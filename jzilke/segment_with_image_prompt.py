import argparse
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch

from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import normalize_bbox


def xywh_to_xyxy(box_xywh):
    x, y, w, h = box_xywh
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def xyxy_to_xywh(box_xyxy):
    x0, y0, x1, y1 = box_xyxy
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def clip_box_xyxy(box_xyxy, image_width: int, image_height: int):
    x0, y0, x1, y1 = box_xyxy
    x0 = float(np.clip(x0, 0, image_width - 1))
    y0 = float(np.clip(y0, 0, image_height - 1))
    x1 = float(np.clip(x1, x0 + 1, image_width))
    y1 = float(np.clip(y1, y0 + 1, image_height))
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def normalize_prompt_box(box_xywh, image_width: int, image_height: int):
    box_tensor = torch.tensor(box_xywh, dtype=torch.float32).view(1, 4)
    box_cxcywh = box_xywh_to_cxcywh(box_tensor)
    normalized_box = normalize_bbox(box_cxcywh, image_width, image_height)
    return normalized_box.flatten().tolist()


def get_best_mask(masks, scores):
    if len(masks) == 0:
        raise RuntimeError("No masks predicted")

    best_idx = int(scores.argmax().item())
    masks_np = masks.squeeze(1).cpu().numpy().astype(np.uint8) * 255
    return best_mask_from_array(masks_np, best_idx)


def best_mask_from_array(masks_np, best_idx: int):
    return masks_np[best_idx]


def save_mask(mask, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mask)
    print(f"Saved mask to: {output_path}")


def save_debug_box(image_bgr, box_xyxy, output_path: Path, label: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = np.round(box_xyxy).astype(int)
    debug_img = image_bgr.copy()
    cv2.rectangle(debug_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.putText(
        debug_img,
        label,
        (x0, max(20, y0 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_path), debug_img)
    print(f"Saved debug box image to: {output_path}")


def crop_reference(image_bgr, reference_box_xywh):
    image_h, image_w = image_bgr.shape[:2]
    ref_box_xyxy = clip_box_xyxy(xywh_to_xyxy(reference_box_xywh), image_w, image_h)
    x0, y0, x1, y1 = np.round(ref_box_xyxy).astype(int)
    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        raise RuntimeError("Reference box produced an empty crop")
    return crop, ref_box_xyxy


def detect_box_with_orb(reference_crop_bgr, target_bgr):
    ref_gray = cv2.cvtColor(reference_crop_bgr, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=2000, fastThreshold=5)
    ref_kp, ref_desc = orb.detectAndCompute(ref_gray, None)
    tgt_kp, tgt_desc = orb.detectAndCompute(target_gray, None)

    if ref_desc is None or tgt_desc is None or len(ref_kp) < 8 or len(tgt_kp) < 8:
        return None, None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = matcher.knnMatch(ref_desc, tgt_desc, k=2)

    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < 0.75 * second.distance:
            good_matches.append(first)

    if len(good_matches) < 8:
        return None, None

    src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([tgt_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if homography is None or inlier_mask is None:
        return None, None

    inlier_count = int(inlier_mask.ravel().sum())
    if inlier_count < 6:
        return None, None

    h, w = ref_gray.shape[:2]
    ref_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    target_corners = cv2.perspectiveTransform(ref_corners, homography).reshape(-1, 2)

    x0 = float(np.min(target_corners[:, 0]))
    y0 = float(np.min(target_corners[:, 1]))
    x1 = float(np.max(target_corners[:, 0]))
    y1 = float(np.max(target_corners[:, 1]))

    return np.array([x0, y0, x1, y1], dtype=np.float32), {
        "method": "orb",
        "score": inlier_count,
    }


def detect_box_with_template_matching(reference_crop_bgr, target_bgr):
    ref_gray = cv2.cvtColor(reference_crop_bgr, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)

    ref_h, ref_w = ref_gray.shape[:2]
    target_h, target_w = target_gray.shape[:2]

    best = None
    scales = np.linspace(0.5, 1.5, 21)
    for scale in scales:
        scaled_w = max(8, int(round(ref_w * scale)))
        scaled_h = max(8, int(round(ref_h * scale)))
        if scaled_w >= target_w or scaled_h >= target_h:
            continue

        resized = cv2.resize(ref_gray, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        result = cv2.matchTemplate(target_gray, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if best is None or max_val > best["score"]:
            x0, y0 = max_loc
            best = {
                "method": "template",
                "score": float(max_val),
                "box_xyxy": np.array(
                    [x0, y0, x0 + scaled_w, y0 + scaled_h],
                    dtype=np.float32,
                ),
            }

    if best is None or best["score"] < 0.35:
        return None, None

    return best["box_xyxy"], {
        "method": best["method"],
        "score": best["score"],
    }


def validate_detected_box(box_xyxy, target_shape, reference_crop_shape):
    target_h, target_w = target_shape[:2]
    ref_h, ref_w = reference_crop_shape[:2]
    box_xyxy = clip_box_xyxy(box_xyxy, target_w, target_h)
    x0, y0, x1, y1 = box_xyxy
    width = x1 - x0
    height = y1 - y0

    if width < 8 or height < 8:
        return None

    scale_x = width / max(ref_w, 1)
    scale_y = height / max(ref_h, 1)
    if scale_x < 0.2 or scale_x > 5.0 or scale_y < 0.2 or scale_y > 5.0:
        return None

    return box_xyxy


def estimate_target_box(reference_crop_bgr, target_bgr):
    orb_box, orb_info = detect_box_with_orb(reference_crop_bgr, target_bgr)
    if orb_box is not None:
        valid_box = validate_detected_box(orb_box, target_bgr.shape, reference_crop_bgr.shape)
        if valid_box is not None:
            return valid_box, orb_info

    template_box, template_info = detect_box_with_template_matching(reference_crop_bgr, target_bgr)
    if template_box is not None:
        valid_box = validate_detected_box(
            template_box, target_bgr.shape, reference_crop_bgr.shape
        )
        if valid_box is not None:
            return valid_box, template_info

    raise RuntimeError(
        "Could not localize the reference object in the target image with ORB or template matching"
    )


def run_sam_with_box(model_processor, target_image_path: Path, target_box_xywh):
    image = Image.open(target_image_path).convert("RGB")
    start_time = time.perf_counter()

    inference_state = model_processor.set_image(image)
    normalized_box = normalize_prompt_box(target_box_xywh, image.width, image.height)
    output = model_processor.add_geometric_prompt(
        state=inference_state,
        box=normalized_box,
        label=True,
    )

    elapsed = time.perf_counter() - start_time
    masks = output.get("masks")
    boxes = output.get("boxes")
    scores = output.get("scores")

    if masks is None or scores is None:
        raise RuntimeError("Inference did not return masks/scores")

    print(f"SAM inference completed in {elapsed:.3f}s")
    return masks, boxes, scores


def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-step exemplar segmentation: match reference crop in target image, then segment with SAM3 box prompt"
    )
    parser.add_argument("--reference_image", required=True, help="Path to the reference image")
    parser.add_argument(
        "--reference_box",
        required=True,
        nargs=4,
        type=float,
        metavar=("X", "Y", "W", "H"),
        help="Reference object box in the reference image as x y width height",
    )
    parser.add_argument("--target_image", required=True, help="Path to the target image")
    parser.add_argument("--save", required=True, help="Path to save output mask image")
    parser.add_argument(
        "--debug_box_image",
        help="Optional path to save the target image with the estimated prompt box drawn on it",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Segmentation threshold")
    return parser.parse_args()


def segmentation(
    reference_image_path: Path,
    target_image_path: Path,
    out_path: Path,
    reference_box_xywh,
    threshold: float = 0.5,
    debug_box_image_path: Path | None = None,
):
    print("Loading images...")
    reference_bgr = cv2.imread(str(reference_image_path), cv2.IMREAD_COLOR)
    target_bgr = cv2.imread(str(target_image_path), cv2.IMREAD_COLOR)
    if reference_bgr is None:
        raise RuntimeError(f"Failed to read reference image: {reference_image_path}")
    if target_bgr is None:
        raise RuntimeError(f"Failed to read target image: {target_image_path}")

    reference_crop_bgr, clipped_reference_box = crop_reference(reference_bgr, reference_box_xywh)

    print(f"Reference image: {reference_image_path}")
    print(f"Reference box (x y w h): {reference_box_xywh}")
    print(f"Target image: {target_image_path}")
    print("Estimating object location in target image...")
    target_box_xyxy, match_info = estimate_target_box(reference_crop_bgr, target_bgr)
    target_box_xywh = xyxy_to_xywh(target_box_xyxy)
    print(
        f"Estimated target box via {match_info['method']} with score {match_info['score']}: "
        f"{[round(v, 1) for v in target_box_xywh]}"
    )

    if debug_box_image_path is not None:
        save_debug_box(
            target_bgr,
            target_box_xyxy,
            debug_box_image_path,
            f"{match_info['method']} {match_info['score']:.3f}",
        )

    print("Loading SAM3 model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=threshold)
    print(f"Confidence threshold: {threshold}")

    masks, _, scores = run_sam_with_box(processor, target_image_path, target_box_xywh)
    best_mask = get_best_mask(masks, scores)
    save_mask(best_mask, out_path)


def main():
    args = parse_args()

    reference_image_path = Path(args.reference_image)
    target_image_path = Path(args.target_image)
    out_path = Path(args.save)
    debug_box_image_path = Path(args.debug_box_image) if args.debug_box_image else None

    if not reference_image_path.is_file():
        raise FileNotFoundError(f"Reference image not found: {reference_image_path}")
    if not target_image_path.is_file():
        raise FileNotFoundError(f"Target image not found: {target_image_path}")

    segmentation(
        reference_image_path=reference_image_path,
        target_image_path=target_image_path,
        out_path=out_path,
        reference_box_xywh=args.reference_box,
        threshold=args.threshold,
        debug_box_image_path=debug_box_image_path,
    )


if __name__ == "__main__":
    main()
