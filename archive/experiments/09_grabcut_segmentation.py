import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

OUTPUT_DIR = "outputs/grabcut"
os.makedirs(OUTPUT_DIR, exist_ok=True)

image_paths = sorted(
    glob.glob("dataset/valid/egg/*.jpg")
)

if not image_paths:
    raise FileNotFoundError("No egg images found.")

image_path = image_paths[0]

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(
        f"Could not read image: {image_path}"
    )

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

print("Image:", image_path)
print("Image dimensions:", image.shape)


# --------------------------------------------------
# 2. Select a rectangle around the egg
# --------------------------------------------------

print("\nDraw a rectangle around the egg.")
print("Keep the rectangle as tight as possible.")
print("Press ENTER or SPACE to confirm.")
print("Press ESC to cancel.")

rectangle = cv2.selectROI(
    "Select Egg Region",
    image,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyAllWindows()

x, y, rect_width, rect_height = rectangle

if rect_width == 0 or rect_height == 0:
    raise RuntimeError(
        "No rectangle selected. Run the script again."
    )


# --------------------------------------------------
# 3. Initialize GrabCut
# --------------------------------------------------

# GrabCut labels:
# 0 = definite background
# 1 = definite foreground
# 2 = probable background
# 3 = probable foreground

mask = np.zeros(
    image.shape[:2],
    dtype=np.uint8
)

background_model = np.zeros(
    (1, 65),
    dtype=np.float64
)

foreground_model = np.zeros(
    (1, 65),
    dtype=np.float64
)

# GrabCut estimates the foreground inside the rectangle.
cv2.grabCut(
    image,
    mask,
    (x, y, rect_width, rect_height),
    background_model,
    foreground_model,
    5,
    cv2.GC_INIT_WITH_RECT
)


# --------------------------------------------------
# 4. Convert GrabCut labels into a binary mask
# --------------------------------------------------

binary_mask = np.where(
    (mask == cv2.GC_FGD)
    | (mask == cv2.GC_PR_FGD),
    255,
    0
).astype(np.uint8)


# --------------------------------------------------
# 5. Optional morphological cleanup
# --------------------------------------------------

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)

cleaned_mask = cv2.morphologyEx(
    binary_mask,
    cv2.MORPH_OPEN,
    kernel
)

cleaned_mask = cv2.morphologyEx(
    cleaned_mask,
    cv2.MORPH_CLOSE,
    kernel
)


# --------------------------------------------------
# 6. Create a visual overlay
# --------------------------------------------------

overlay = image_rgb.copy()

foreground_pixels = cleaned_mask > 0

overlay[foreground_pixels] = (
    0.5 * overlay[foreground_pixels].astype(np.float32)
    + 0.5 * np.array(
        [255, 0, 0],
        dtype=np.float32
    )
).astype(np.uint8)


# --------------------------------------------------
# 7. Save outputs
# --------------------------------------------------

mask_path = os.path.join(
    OUTPUT_DIR,
    "grabcut_mask.png"
)

overlay_path = os.path.join(
    OUTPUT_DIR,
    "grabcut_overlay.jpg"
)

comparison_path = os.path.join(
    OUTPUT_DIR,
    "grabcut_comparison.png"
)

cv2.imwrite(
    mask_path,
    cleaned_mask
)

cv2.imwrite(
    overlay_path,
    cv2.cvtColor(
        overlay,
        cv2.COLOR_RGB2BGR
    )
)

print("\nGrabCut completed.")
print("Foreground pixels:", np.count_nonzero(cleaned_mask))
print("Saved mask:", mask_path)
print("Saved overlay:", overlay_path)


# --------------------------------------------------
# 8. Display the result
# --------------------------------------------------

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cleaned_mask, cmap="gray")
plt.title("GrabCut Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title("GrabCut Overlay")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    comparison_path,
    dpi=150
)

plt.show()

print("Saved comparison:", comparison_path)
