import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Find the original image
image_paths = glob.glob("dataset/valid/egg/*.jpg")

if not image_paths:
    raise FileNotFoundError("No dataset images found.")

image = cv2.imread(image_paths[0])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Find saved SAM masks
mask_paths = sorted(
    glob.glob("outputs/sam/mask_*.png")
)

if not mask_paths:
    raise FileNotFoundError(
        "No saved masks found. Run step10_sam.py first."
    )

image_height, image_width = image.shape[:2]
image_area = image_height * image_width

print(f"Image dimensions: {image_width} x {image_height}")
print(f"Total pixels: {image_area}")
print(f"Number of masks: {len(mask_paths)}")
print()

for mask_path in mask_paths:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        continue

    binary_mask = mask > 0
    area = int(np.count_nonzero(binary_mask))
    area_ratio = area / image_area

    ys, xs = np.where(binary_mask)

    if area > 0:
        x_min = int(xs.min())
        x_max = int(xs.max())
        y_min = int(ys.min())
        y_max = int(ys.max())

        bbox = (
            x_min,
            y_min,
            x_max - x_min + 1,
            y_max - y_min + 1,
        )
    else:
        bbox = None

    print(os.path.basename(mask_path))
    print(f"  Area: {area} pixels")
    print(f"  Image area ratio: {area_ratio:.3f}")
    print(f"  Bounding box (x, y, width, height): {bbox}")

    if area > 0:
        print(f"  Bounding-box width: {bbox[2]}")
        print(f"  Bounding-box height: {bbox[3]}")

    print()
