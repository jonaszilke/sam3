from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import numpy as np
from typing import Any, Dict, Optional
import logging
import time

logger = logging.getLogger(__name__)

def _as_numpy_image(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    if array.ndim not in (2, 3):
        raise ValueError(f"{name} must be HxW or HxWxC, got shape {array.shape}")
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)
    return array


def _to_pil_image(image: np.ndarray, image_format: str) -> Image.Image:
    if image.ndim == 2:
        return Image.fromarray(image, mode="L")

    if image.shape[2] == 4:
        if image_format == "bgra":
            image = image[..., [2, 1, 0, 3]]
        elif image_format != "rgba":
            raise ValueError(f"unsupported image_format for 4-channel image: {image_format}")
        return Image.fromarray(image, mode="RGBA")

    if image.shape[2] != 3:
        raise ValueError(f"expected 3 or 4 channels, got shape {image.shape}")

    if image_format == "bgr":
        image = image[..., ::-1]
    elif image_format != "rgb":
        raise ValueError(f"unsupported image_format: {image_format}")

    return Image.fromarray(image, mode="RGB")



class Sam3Segmenter:
    def __init__(
        self,
        prompt: Optional[str],
        confidence_threshold: float,
    ) -> None:
        self.prompt = prompt
        self.confidence_threshold = float(confidence_threshold)
        self.frame_index = 0

        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model, confidence_threshold=self.confidence_threshold)
        logger.info("SAM3 model initialized")

    def status(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "confidence_threshold": self.confidence_threshold,
            "frame_index": self.frame_index,
        }

    def set_prompt(self, prompt: str) -> Dict[str, Any]:
        if not prompt:
            raise ValueError("prompt must be a non-empty string")
        self.prompt = prompt
        logger.info("Updated default prompt: %s", self.prompt)
        return {"prompt": self.prompt}

    def set_confidence_threshold(self, new_confidence_threshold: float) -> Dict[str, Any]:
        new_confidence_threshold = float(new_confidence_threshold)
        if new_confidence_threshold != self.confidence_threshold:
            self.confidence_threshold = new_confidence_threshold
            self.processor.set_confidence_threshold(self.confidence_threshold)
            logger.info("Updated confidence threshold to %.3f", self.confidence_threshold)
        return {"confidence_threshold": self.confidence_threshold}

    def _run_inference(self, image: np.ndarray, prompt: str, image_format: str) -> Dict[str, Any]:
        pil_image = _to_pil_image(image, image_format=image_format)

        start_time = time.perf_counter()
        inference_state = self.processor.set_image(pil_image)
        output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)
        elapsed = time.perf_counter() - start_time

        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")
        if masks is None or scores is None:
            raise RuntimeError("Inference did not return masks/scores")
        if len(masks) == 0:
            raise RuntimeError("No masks predicted")

        best_idx = int(scores.argmax().item())
        masks_np = masks.squeeze(1).detach().cpu().numpy().astype(np.uint8) * 255
        scores_np = scores.detach().cpu().numpy()
        boxes_np = None if boxes is None else boxes.detach().cpu().numpy()
        best_box = None if boxes_np is None or len(boxes_np) <= best_idx else np.asarray(boxes_np[best_idx], dtype=np.float32)

        return {
            "mask": masks_np[best_idx],
            "score": float(scores_np[best_idx]),
            "box": best_box,
            "num_masks": int(len(masks_np)),
            "elapsed_sec": elapsed,
        }

    def segment_image(
        self,
        image: np.ndarray,
        image_format: str = "bgr",
        prompt: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        prompt = prompt or self.prompt
        if not prompt:
            raise ValueError("segment_image requires prompt or a prior start_segmentation/set_prompt call")

        if confidence_threshold is not None:
            self.set_confidence_threshold(confidence_threshold)

        image = _as_numpy_image(image, "image")
        image_format = str(image_format).lower()
        result = self._run_inference(image=image, prompt=prompt, image_format=image_format)

        self.frame_index += 1
        return {
            "frame_index": self.frame_index,
            "prompt": prompt,
            "confidence_threshold": self.confidence_threshold,
            "score": result["score"],
            "box": result["box"],
            "num_masks": result["num_masks"],
            "elapsed_sec": result["elapsed_sec"],
            "mask": result["mask"],
        }
