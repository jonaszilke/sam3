import pyrealsense2 as rs
import numpy as np
import cv2
import time

pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(cfg)
print("Press ESC to exit; s/space to save frame.")

try:
    i = 0
    while True:
        try:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
        except Exception as e:
            print(f"Warning: frame timeout or error: {e}")
            continue

        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            print("Warning: no color or depth frame, retrying...")
            continue

        col = np.asanyarray(color.get_data())
        dep = np.asanyarray(depth.get_data())
        dep_vis = cv2.applyColorMap(cv2.convertScaleAbs(dep, alpha=0.03), cv2.COLORMAP_JET)

        cv2.imshow("color", col)
        cv2.imshow("depth", dep_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord("s") or key == 32:
            cv2.imwrite(f"test_color_{i:03d}.png", col)
            cv2.imwrite(f"test_depth_{i:03d}.png", dep)
            print(f"saved test_color_{i:03d}.png, test_depth_{i:03d}.png")
            i += 1
finally:
    pipeline.stop()
    cv2.destroyAllWindows()