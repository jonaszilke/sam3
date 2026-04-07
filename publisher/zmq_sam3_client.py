#!/usr/bin/env python3
"""Small client that drives the ZeroMQ SAM3 request/reply server."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import zmq


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vis = image.copy()
    positive = mask > 0
    vis[positive] = (0.4 * vis[positive] + 0.6 * np.array([0, 255, 0], dtype=np.float32)).astype(np.uint8)
    return vis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Client for the ZeroMQ SAM3 publisher")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--prompt", required=True, help="Text prompt for segmentation")
    parser.add_argument("--rep_endpoint", type=str, default="tcp://127.0.0.1:5565")
    parser.add_argument("--confidence_threshold", type=float, default=0.2)
    parser.add_argument("--save_mask", type=str, default=None, help="Optional path to save the predicted mask")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image = cv2.imread(str(Path(args.image)))
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {args.image}")

    context = zmq.Context()
    req_socket = context.socket(zmq.REQ)
    req_socket.connect(args.rep_endpoint)

    try:
        req_socket.send_pyobj(
            {
                "command": "segment_image",
                "image": image,
                "image_format": "bgr",
                "prompt": args.prompt,
                "confidence_threshold": args.confidence_threshold,

            }
        )
        reply = req_socket.recv_pyobj()
        if reply.get("status") != "ok":
            raise RuntimeError(f"segment_image failed: {reply}")

        result = reply["result"]
        mask = np.asarray(result["mask"], dtype=np.uint8)

        if args.save_mask:
            cv2.imwrite(args.save_mask, mask)
        else:
            vis = overlay_mask(image, mask)
            box = result.get("box")
            if box is not None and len(box) == 4:
                x1, y1, x2, y2 = np.asarray(box, dtype=np.int32).tolist()
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

            label = f"score={result['score']:.3f}"
            cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.imshow("SAM3 ZeroMQ Client", vis)
            cv2.waitKey(0)
    finally:
        # if not args.save_mask:
        cv2.destroyAllWindows()
        req_socket.close(0)
        context.term()


if __name__ == "__main__":
    main()
