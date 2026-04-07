import argparse
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


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

    print(f"Inference completed in {elapsed:.3f}s")
    return masks, boxes, scores


def get_best_mask(masks, scores):
    if len(masks) == 0:
        raise RuntimeError("No masks predicted")

    best_idx = int(scores.argmax().item())
    masks_np = masks.squeeze(1).cpu().numpy().astype(np.uint8) * 255
    best_mask = masks_np[best_idx]
    return best_mask


def save_mask(mask, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mask)
    print(f"Saved mask to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 text-guided segmentation for one image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--save", required=True, help="Path to save output mask image (PNG recommended)")
    parser.add_argument("--prompt", required=True, help="Text prompt for segmentation")
    parser.add_argument("--threshold", type=float, default=0.5, help="Segmentation Threshold")

    return parser.parse_args()


def segmentation(img_path: Path, out_path: Path, prompt: str, threshold: float=0.5):
    print("Loading SAM3 model...")
    model = build_sam3_image_model()
    print(f"Confidence threshold: {threshold}")
    processor = Sam3Processor(model, confidence_threshold=threshold)

    print(f"Image: {img_path}")
    print(f"Prompt: {prompt}")

    masks, boxes, scores = run_sam(processor, img_path, prompt)
    best_mask = get_best_mask(masks, scores)

    save_mask(best_mask, out_path)

def main():
    args = parse_args()
    img_path = Path(args.image)
    out_path = Path(args.save)

    if not img_path.is_file():
        raise FileNotFoundError(f"Input image not found: {img_path}")

    segmentation(img_path, out_path, prompt=args.prompt, threshold=args.threshold)





if __name__ == "__main__":
    main()