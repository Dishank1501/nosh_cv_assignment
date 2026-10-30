import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# Find the original image
image_paths = glob.glob("dataset/valid/egg/*.jpg")

if not image_paths:
    raise FileNotFoundError("No images found.")

image_path = image_paths[0]

image_bgr = cv2.imread(image_path)

if image_bgr is None:
    raise FileNotFoundError(f"Could not read {image_path}")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Load the candidate mask that appears to represent the egg
mask_path = "outputs/sam/mask_004.png"

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if mask is None:
    raise FileNotFoundError(f"Could not read {mask_path}")

# Convert the mask into a Boolean array
binary_mask = mask > 0

# Create an overlay
overlay = image_rgb.copy()

# Tint selected pixels red for visibility
overlay[binary_mask] = (
    0.5 * overlay[binary_mask]
    + 0.5 * np.array([255, 0, 0])
).astype(np.uint8)

# Display results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(binary_mask, cmap="gray")
plt.title("Selected SAM Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title("Egg Mask Overlay")
plt.axis("off")

plt.tight_layout()

# Save the result
os.makedirs("outputs/sam", exist_ok=True)

plt.savefig(
    "outputs/sam/selected_egg_mask.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("Selected mask:", mask_path)
print("Mask area:", int(np.count_nonzero(binary_mask)), "pixels")
print("Saved visualization: outputs/sam/selected_egg_mask.png")

