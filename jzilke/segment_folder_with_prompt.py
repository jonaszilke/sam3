import argparse
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import tqdm

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def run_sam(model_processor, img_path: Path, prompt: str):
    image = Image.open(img_path).convert("RGB")
    start_time = time.perf_counter()

    inference_state = model_processor.set_image(image)
    output = model_processor.set_text_prompt(state=inference_state, prompt=prompt)

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    masks = output.get("masks")
    boxes = output.get("boxes")
    scores = output.get("scores")

    if masks is None or scores is None:
        raise RuntimeError("Inference did not return masks/scores")

    # print(f"Inference completed in {elapsed:.3f}s")
    return masks, boxes, scores


def get_best_mask(masks, scores):
    if len(masks) == 0:
        return None

    best_idx = int(scores.argmax().item())
    masks_np = masks.squeeze(1).cpu().numpy().astype(np.uint8) * 255
    best_mask = masks_np[best_idx]
    return best_mask


def save_mask(mask, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mask)
    # print(f"Saved mask to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 text-guided segmentation for all images in a folder")
    parser.add_argument("--input_dir", required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", required=True, help="Directory to save output mask images")
    parser.add_argument("--prompt", required=True, help="Text prompt for segmentation")
    return parser.parse_args()


def get_image_paths(input_dir: Path):
    image_paths = [
        path for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_paths:
        raise RuntimeError(f"No supported image files found in: {input_dir}")

    return image_paths


def segment_folder(input_dir: Path, output_dir: Path, prompt: str):
    print("Loading SAM3 model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=0.2)

    image_paths = get_image_paths(input_dir)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Prompt: {prompt}")
    print(f"Found {len(image_paths)} image(s)")

    for img_path in tqdm.tqdm(image_paths):
        out_path = output_dir / img_path.name
        # print(f"Processing: {img_path}")

        masks, boxes, scores = run_sam(processor, img_path, prompt)
        best_mask = get_best_mask(masks, scores)

        if best_mask is None:
            print(f"No mask found for: {img_path}, skipping")
            continue

        save_mask(best_mask, out_path)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    segment_folder(input_dir, output_dir, prompt=args.prompt)


if __name__ == "__main__":
    main()
