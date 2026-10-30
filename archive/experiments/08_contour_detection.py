import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# Configuration
# --------------------------------------------------

OUTPUT_DIR = "outputs/classical_contours"
os.makedirs(OUTPUT_DIR, exist_ok=True)

image_paths = sorted(glob.glob("dataset/valid/egg/*.jpg"))

if not image_paths:
    raise FileNotFoundError("No egg images found.")

# Start with the same first validation image as your old script.
image_path = image_paths[0]

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not read {image_path}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

height, width = gray.shape
image_area = height * width


# --------------------------------------------------
# Step 1: Reduce image noise
# --------------------------------------------------

blurred = cv2.GaussianBlur(gray, (5, 5), 0)


# --------------------------------------------------
# Step 2: Test both Otsu threshold polarities
# --------------------------------------------------

threshold_value, mask_bright = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
)

_, mask_dark = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
)

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5),
)


def get_contours(binary_mask):
    """Clean a binary mask and return its external contours."""

    opened = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    closed = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel,
    )

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    return closed, contours


# --------------------------------------------------
# Step 3: Find candidate contours
# --------------------------------------------------

candidates = []

for polarity_name, binary_mask in [
    ("bright", mask_bright),
    ("dark", mask_dark),
]:
    cleaned_mask, contours = get_contours(binary_mask)

    for contour in contours:
        area = cv2.contourArea(contour)
        area_ratio = area / image_area

        # Experimental area filter; tune after inspecting results.
        if area_ratio < 0.02 or area_ratio > 0.40:
            continue

        x, y, box_width, box_height = cv2.boundingRect(contour)

        if box_width == 0 or box_height == 0:
            continue

        aspect_ratio = box_width / box_height

        # Compactness: 1 is ideal for a circle; lower values
        # indicate more irregular shapes.
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        circularity = (
            4 * np.pi * area / (perimeter * perimeter)
        )

        # Normalize distance from image centre.
        center_x = x + box_width / 2
        center_y = y + box_height / 2

        center_distance = np.sqrt(
            ((center_x - width / 2) / width) ** 2
            + ((center_y - height / 2) / height) ** 2
        )

        center_score = max(0.0, 1.0 - center_distance)

        # Experimental heuristic, not a probability.
        shape_score = (
            1.0 if 0.7 <= aspect_ratio <= 2.2 else 0.0
        )

        score = (
            0.45 * shape_score
            + 0.30 * circularity
            + 0.25 * center_score
        )

        candidates.append({
            "contour": contour,
            "score": score,
            "area": area,
            "polarity": polarity_name,
            "mask": cleaned_mask,
            "bbox": (x, y, box_width, box_height),
            "aspect_ratio": aspect_ratio,
            "circularity": circularity,
        })


# --------------------------------------------------
# Step 4: Select the highest-ranked contour
# --------------------------------------------------

best_candidate = None

if candidates:
    best_candidate = max(
        candidates,
        key=lambda item: item["score"],
    )


# --------------------------------------------------
# Step 5: Save and display results
# --------------------------------------------------

if best_candidate is None:
    print("No contour passed the experimental filters.")
    print("Try inspecting both threshold masks.")

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "otsu_bright.png"),
        mask_bright,
    )

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "otsu_dark.png"),
        mask_dark,
    )

    raise SystemExit(0)


selected_mask = np.zeros(
    (height, width),
    dtype=np.uint8,
)

cv2.drawContours(
    selected_mask,
    [best_candidate["contour"]],
    -1,
    255,
    thickness=cv2.FILLED,
)

overlay = image_rgb.copy()
foreground = selected_mask > 0

overlay[foreground] = (
    0.5 * overlay[foreground].astype(np.float32)
    + 0.5 * np.array([255, 0, 0], dtype=np.float32)
).astype(np.uint8)

mask_path = os.path.join(
    OUTPUT_DIR,
    "selected_contour_mask.png",
)

overlay_path = os.path.join(
    OUTPUT_DIR,
    "selected_contour_overlay.jpg",
)

cv2.imwrite(mask_path, selected_mask)

cv2.imwrite(
    overlay_path,
    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
)

# Save intermediate masks for debugging.
cv2.imwrite(
    os.path.join(OUTPUT_DIR, "otsu_bright.png"),
    mask_bright,
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "otsu_dark.png"),
    mask_dark,
)

print("Image:", image_path)
print("Otsu threshold:", threshold_value)
print("Candidate contours:", len(candidates))
print("Selected polarity:", best_candidate["polarity"])
print("Selected contour area:", best_candidate["area"])
print("Selected aspect ratio:", best_candidate["aspect_ratio"])
print("Selected circularity:", best_candidate["circularity"])
print("Selected score:", best_candidate["score"])
print("Saved mask:", mask_path)
print("Saved overlay:", overlay_path)


plt.figure(figsize=(15, 8))

plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(mask_bright, cmap="gray")
plt.title("Otsu: Bright Foreground")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(selected_mask, cmap="gray")
plt.title("Selected Contour Mask")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(overlay)
plt.title("Selected Contour Overlay")
plt.axis("off")

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "contour_comparison.png"),
    dpi=150,
)
plt.show()