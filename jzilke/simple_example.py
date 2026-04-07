import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import numpy as np
import cv2
import time

import random
from pathlib import Path


def get_files_from_dir(directory_path):
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        raise ValueError(f"The path {directory_path} is not a valid directory.")

    files = [f for f in dir_path.iterdir() if f.is_file()]
    return files

def get_random_files(directory_path, n):
    files = get_files_from_dir(directory_path)
    sample_size = min(len(files), n)
    return random.sample(files, sample_size)

model = build_sam3_image_model()
processor = Sam3Processor(model)

# Load the model
def run_sam(img_path, prompt):
    image = Image.open(img_path).convert("RGB")

    start_time = time.perf_counter()
    inference_state = processor.set_image(image)
    # Prompt the model with text
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)

    end_time = time.perf_counter()

    print(f"Execution time: {end_time - start_time:.4f} seconds")

    # Get the masks, bounding boxes, and scores
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    return masks, boxes, scores

def get_best_mask(img_path, prompt):
    image = Image.open(img_path).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    masks, boxes, scores = run_sam(img_path, prompt)

    if len(masks) == 0:
        raise RuntimeError("No mask found")

    try:
        best_idx = scores.argmax().item()
    except IndexError:
        return image_bgr

    masks_np = masks.squeeze(1).cpu().numpy().astype(np.uint8) * 255
    best_mask = masks_np[best_idx]
    return best_mask


def save_best_masks_in_directory(directory_path):
    files = get_files_from_dir(directory_path)

    bgr_imgs = []
    mask_imgs = []

    for file_path in files:
        try:
            prompt = "electrical connector"
            best_mask = get_best_mask(file_path, prompt)

            mask_path = file_path.with_name(f"{file_path.stem}_mask.png")
            cv2.imwrite(str(mask_path), best_mask)
            print(f"Saved {mask_path}")

            # collect for visualization
            bgr = cv2.imread(str(file_path))
            if bgr is None:
                print(f"Warning: failed to load {file_path}")
                continue
            bgr_imgs.append(bgr)
            mask_imgs.append(best_mask)

        except Exception as e:
            print(f"Skipping {file_path}: {e}")

    if len(bgr_imgs) == 0:
        print("No images to display.")
        return

    processed_bgr = []
    processed_masks = []

    display_size = (320, 240)

    for i in range(len(bgr_imgs)):
        img = cv2.resize(bgr_imgs[i], display_size)

        m = mask_imgs[i]
        if len(m.shape) == 2:
            m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        m = cv2.resize(m, display_size)

        processed_bgr.append(img)
        processed_masks.append(m)

    top_row = np.hstack(processed_bgr)
    bottom_row = np.hstack(processed_masks)
    full_grid = np.vstack((top_row, bottom_row))

    cv2.imshow("RGB (Top) vs Mask (Bottom)", full_grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



img_dir = "/images/"

save_best_masks_in_directory(img_dir)

exit()

# files = get_random_files(img_dir, n=10)
mask_imgs = []
bgr_imgs = []
# for f in files:
#     bgr, dept = run_sam(f)
#     bgr_imgs.append(bgr)
#     mask_imgs.append(dept)

mask = get_best_mask('/images/Screenshot.png')
bgr_imgs.append(bgr)
mask_imgs.append(mask)
processed_bgr = []
processed_depth = []

for i in range(len(bgr_imgs)):
    # 1. Ensure BGR is a numpy array (if it's still PIL)
    img_bgr = np.array(bgr_imgs[i])

    # Apply a colormap so depth features are visible
    m = mask_imgs[i].copy()

    # 1. Normalize depth to 0-255 range (8-bit)
    d_norm = cv2.normalize(mask_imgs[i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    try:
        d_colored = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    except cv2.error:
        d_colored = m

    # 3. Optional: Resize if they are too big for your screen
    # e.g., resize to 320x240 for a manageable grid
    display_size = (4*320, 4*240)
    img_bgr = cv2.resize(img_bgr, display_size)
    processed_bgr.append(img_bgr)

    d_colored = cv2.resize(d_colored, display_size)
    processed_depth.append(d_colored)

top_row = np.hstack(processed_bgr)
bottom_row = np.hstack(processed_depth)

# Combine them vertically
full_grid = np.vstack((top_row, bottom_row))

# Show the result
cv2.imshow("SAM Results: RGB (Top) vs Mask (Bottom)", full_grid)
cv2.waitKey(0)
cv2.destroyAllWindows()
