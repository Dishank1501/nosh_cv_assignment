import os
import glob
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator,
)

# -----------------------------------------------
# 1. Paths and configuration
# -----------------------------------------------

CHECKPOINT = "models/sam_vit_b_01ec64.pth"
OUTPUT_DIR = "outputs/selected_egg"

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_paths = glob.glob("dataset/valid/egg/*.jpg")

if not image_paths:
    raise FileNotFoundError("No egg images found.")

image_path = image_paths[0]
print("Processing:", image_path)

# -----------------------------------------------
# 2. Load image
# -----------------------------------------------

image_bgr = cv2.imread(image_path)

if image_bgr is None:
    raise FileNotFoundError(f"Could not read {image_path}")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

height, width = image_rgb.shape[:2]
image_area = height * width

# -----------------------------------------------
# 3. Load SAM
# -----------------------------------------------

torch.set_num_threads(4)

sam = sam_model_registry["vit_b"](
    checkpoint=CHECKPOINT
)

sam.to(device="cpu")
sam.eval()

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=8,
    points_per_batch=4,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.90,
    crop_n_layers=0,
    min_mask_region_area=100,
)

# -----------------------------------------------
# 4. Generate candidate masks
# -----------------------------------------------

print("Generating SAM masks...")

with torch.inference_mode():
    masks = mask_generator.generate(image_rgb)

print("Candidates generated:", len(masks))

if not masks:
    raise RuntimeError("SAM did not generate any masks.")

# -----------------------------------------------
# 5. Rank candidates using geometry
# -----------------------------------------------

image_cx = width / 2
image_cy = height / 2

best_mask = None
best_score = float("-inf")

for index, candidate in enumerate(masks):
    segmentation = candidate["segmentation"]
    area = int(np.count_nonzero(segmentation))

    if area == 0:
        continue

    x, y, box_width, box_height = candidate["bbox"]

    area_ratio = area / image_area

    box_cx = x + box_width / 2
    box_cy = y + box_height / 2

    # Distance from the image center, normalized to [approximately 0, 1]
    center_distance = np.sqrt(
        ((box_cx - image_cx) / width) ** 2
        + ((box_cy - image_cy) / height) ** 2
    )

    # Favor reasonably sized regions rather than tiny details
    if area_ratio < 0.02 or area_ratio > 0.40:
        continue

    # Favor regions close to the image center
    score = (
        1.0
        - center_distance
    )

    # Slightly favor regions with a plausible oval-like bounding box
    aspect_ratio = box_width / max(box_height, 1)

    if 1.1 <= aspect_ratio <= 2.2:
        score += 0.15

    # Store the best candidate
    if score > best_score:
        best_score = score
        best_mask = segmentation
        best_index = index

if best_mask is None:
    raise RuntimeError(
        "No candidate passed the geometric filters."
    )

print("Selected candidate index:", best_index)
print("Selection score:", round(best_score, 3))

# -----------------------------------------------
# 6. Save selected binary mask
# -----------------------------------------------

selected_mask = best_mask.astype(np.uint8) * 255

mask_output_path = os.path.join(
    OUTPUT_DIR,
    "selected_egg_mask.png",
)

cv2.imwrite(mask_output_path, selected_mask)

# -----------------------------------------------
# 7. Create an overlay
# -----------------------------------------------

overlay = image_rgb.copy()

foreground = best_mask.astype(bool)

red = np.zeros_like(overlay)
red[:, :, 0] = 255

overlay[foreground] = (
    0.5 * overlay[foreground]
    + 0.5 * red[foreground]
).astype(np.uint8)

# -----------------------------------------------
# 8. Save visualization
# -----------------------------------------------

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(best_mask, cmap="gray")
plt.title("Automatically Selected Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title("Selected Mask Overlay")
plt.axis("off")

plt.tight_layout()

comparison_path = os.path.join(
    OUTPUT_DIR,
    "comparison.png",
)

plt.savefig(
    comparison_path,
    dpi=150,
    bbox_inches="tight",
)

plt.show()

print("Saved mask:", mask_output_path)
print("Saved comparison:", comparison_path)
