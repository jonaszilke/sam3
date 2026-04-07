import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import pyrealsense2 as rs


def load_sam():
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model()
    return Sam3Processor(model, confidence_threshold=0.2)


def parse_args():
    parser = argparse.ArgumentParser(description="RealSense capture + SAM segmentation")
    parser.add_argument("--save-folder", required=True, help="Directory where rgb/depth/segmentation are saved")
    parser.add_argument("--prompt", required=True, help="Text prompt for SAM segmentation")
    parser.add_argument("--width", type=int, default=640, help="Color frame width")
    parser.add_argument("--height", type=int, default=480, help="Color frame height")
    parser.add_argument("--fps", type=int, default=30, help="Camera frame rate")
    return parser.parse_args()


def init_realsense(width, height, fps):
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(cfg)

    # depth_sensor = profile.get_device().first_depth_sensor()
    # if depth_sensor.supports(rs.option.emitter_enabled):
    #     depth_sensor.set_option(rs.option.emitter_enabled, 1)
    return pipeline, cfg


def run_sam(processor, color_image_bgr, prompt):
    image_rgb = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)

    inference_state = processor.set_image(pil_img)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)

    masks = output.get("masks")
    scores = output.get("scores")
    if masks is None or scores is None or len(masks) == 0:
        raise RuntimeError("SAM did not return masks for prompt")

    best_idx = int(scores.argmax().item())
    masks_np = masks.squeeze(1).cpu().numpy().astype(np.uint8) * 255
    return masks_np[best_idx]


def save_snapshot(out_dir: Path, idx: int, rgb_bgr, depth_raw, mask):
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = out_dir / f"rgb_{idx:04d}.png"
    depth_path = out_dir / f"depth_{idx:04d}.png"
    depth_vis_path = out_dir / f"depth_{idx:04d}_vis.png"
    seg_path = out_dir / f"seg_{idx:04d}.png"

    cv2.imwrite(str(rgb_path), rgb_bgr)
    cv2.imwrite(str(depth_path), depth_raw)

    depth_vis = cv2.convertScaleAbs(depth_raw, alpha=0.03)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    cv2.imwrite(str(depth_vis_path), depth_vis)

    cv2.imwrite(str(seg_path), mask)

    return rgb_path, depth_path, depth_vis_path, seg_path


def main():
    args = parse_args()
    save_folder = Path(args.save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)





    print("Initializing RealSense camera")
    pipeline, cfg = init_realsense(args.width, args.height, args.fps)
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    combined = np.hstack([color_image, depth_colored])
    cv2.putText(combined, "SPACE/s: capture, ESC: exit", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("RealSense RGB+Depth", combined)
    while True:
        cv2.imshow("RealSense RGB+Depth", combined)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    print("Load SAM3")
    sam_processor = load_sam()

    frame_id = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            combined = np.hstack([color_image, depth_colored])
            cv2.putText(combined, "SPACE/s: capture, ESC: exit", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("RealSense RGB+Depth", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

            if key == ord("s") or key == 32:  # save
                frame_id += 1
                timestamp = int(time.time())
                sample_dir = save_folder / f"capture_{timestamp}_{frame_id:03d}"

                print(f"Capture {frame_id}: running segmentation...")
                try:
                    pass
                    best_mask = run_sam(sam_processor, color_image, args.prompt)
                except Exception as e:
                    print(f"Segmentation failed: {e}")
                    continue

                rgb_path, depth_path, depth_vis_path, seg_path = save_snapshot(sample_dir, frame_id, color_image, depth_image, best_mask)
                print(f"Saved: {rgb_path}, {depth_path}, {depth_vis_path}, {seg_path}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()