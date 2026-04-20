#!/usr/bin/env python3
"""ZeroMQ REP server for SAM3 text-guided image segmentation."""

from __future__ import annotations

import argparse
import logging
import random
import traceback
from typing import Any, Dict, Optional

import numpy as np
import zmq

from . import Sam3Segmenter


logger = logging.getLogger(__name__)


def set_logging_format() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


class ZmqSam3Publisher:
    def __init__(
        self,
        segmenter: Sam3Segmenter,
        rep_endpoint: str,
        context: Optional[zmq.Context] = None,
    ) -> None:
        self.segmenter = segmenter
        self.rep_endpoint = rep_endpoint

        self.context = context or zmq.Context()
        self.rep_socket = self.context.socket(zmq.REP)
        self.rep_socket.bind(self.rep_endpoint)

    def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(message, dict):
            raise ValueError("command message must be a dict")

        command = message.get("command")
        if not command:
            raise ValueError("command is required")

        if command == "ping":
            return {
                "status": "ok",
                "command": command,
                "result": self.segmenter.status(),
            }
        if command == "set_prompt":
            result = self.segmenter.set_prompt(message.get("prompt"))
        elif command == "set_confidence_threshold":
            result = self.segmenter.set_confidence_threshold(message.get("confidence_threshold"))
        elif command == "segment_image":
            result = self.segmenter.segment_image(
                image=message.get("image"),
                image_format=message.get("image_format", "bgr"),
                prompt=message.get("prompt"),
                confidence_threshold=message.get("confidence_threshold"),
            )
        elif command == "shutdown":
            result = {"message": "shutting down"}
        else:
            raise ValueError(f"unknown command: {command}")

        return {"status": "ok", "command": command, "result": result}

    def serve_forever(self) -> None:
        logger.info("REP socket bound to %s", self.rep_endpoint)
        try:
            while True:
                message = self.rep_socket.recv_pyobj()
                try:
                    response = self.handle_message(message)
                except Exception as exc:
                    logger.exception("Command failed")
                    response = {
                        "status": "error",
                        "command": message.get("command") if isinstance(message, dict) else None,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                self.rep_socket.send_pyobj(response)
                if response.get("status") == "ok" and response.get("command") == "shutdown":
                    break
        finally:
            self.rep_socket.close(0)
            self.context.term()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZeroMQ SAM3 REP server")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--confidence_threshold", type=float, default=0.2)
    parser.add_argument("--rep_endpoint", type=str, default="tcp://127.0.0.1:5565")
    # parser.add_argument("--rep_endpoint", type=str, default="tcp://0.0.0.0:5565") # public
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_logging_format()
    set_seed(0)

    segmenter = Sam3Segmenter(
        prompt=args.prompt,
        confidence_threshold=args.confidence_threshold,
    )
    server = ZmqSam3Publisher(
        segmenter=segmenter,
        rep_endpoint=args.rep_endpoint,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
