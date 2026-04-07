import cv2
import numpy as np
import pyrealsense2 as rs
import time
import os

SAVE_DIR = '/home/jzilke/ws/sam3/images'

# Configure pipeline
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow("RGB Image", color_image)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        if key == ord('s') or key == 32:  # 's' or SPACE
            filename = f"image_{int(time.time())}.png"
            save_path = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(save_path, color_image)
            print(f"Saved {filename}")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()



