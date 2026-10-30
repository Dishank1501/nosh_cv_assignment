import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor

CHECKPOINT = "models/sam_vit_b_01ec64.pth"

IMAGE_PATH = (
    "dataset/valid/egg/"
    "not_damaged_102_jpg.rf.8d754cbc174c6cb6fb1b5c1929edcbb1.jpg"
)

OUTPUT_DIR = "outputs/positive_negative"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load the image
image_bgr = cv2.imread(IMAGE_PATH)

if image_bgr is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 2. Load SAM
torch.set_num_threads(4)

print("Loading SAM...")
sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
sam.to(device="cpu")
sam.eval()

predictor = SamPredictor(sam)

print("Preparing image for SAM...")
predictor.set_image(image_rgb)

# 3. Collect two points
clicked_points = []

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(image_rgb)
ax.set_title("Click inside the egg, then on the hand/background")
ax.axis("off")


def on_click(event):
    if (
        event.inaxes != ax
        or event.xdata is None
        or event.ydata is None
    ):
        return

    if len(clicked_points) >= 2:
        return

    x = int(event.xdata)
    y = int(event.ydata)

    clicked_points.append([x, y])

    if len(clicked_points) == 1:
        ax.scatter(x, y, c="lime", s=100, marker="x")
        ax.set_title("Now click on the hand/background")
        print(f"Positive point: ({x}, {y})")

    else:
        ax.scatter(x, y, c="red", s=100, marker="x")
        ax.set_title("Points selected")
        print(f"Negative point: ({x}, {y})")
        plt.close(fig)

    fig.canvas.draw_idle()


fig.canvas.mpl_connect("button_press_event", on_click)
plt.show()

if len(clicked_points) != 2:
    raise RuntimeError(
        "Two points are required: one inside the egg and one outside."
    )

# Label 1 = foreground; label 0 = background
point_coords = np.array(clicked_points, dtype=np.float32)
point_labels = np.array([1, 0], dtype=np.int32)

# 4. Generate masks
print("Generating masks...")

with torch.inference_mode():
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

# 5. Select the highest-scoring candidate
best_index = int(np.argmax(scores))
best_mask = masks[best_index]

print("Candidate scores:", scores)
print("Selected candidate:", best_index)
print("Predicted quality score:", round(float(scores[best_index]), 4))
print("Mask area:", int(np.count_nonzero(best_mask)))

# 6. Save the binary mask
mask_path = os.path.join(OUTPUT_DIR, "egg_mask.png")

cv2.imwrite(
    mask_path,
    best_mask.astype(np.uint8) * 255,
)

# 7. Create the overlay
overlay = image_rgb.copy()
overlay[best_mask] = (
    0.5 * overlay[best_mask]
    + 0.5 * np.array([255, 0, 0])
).astype(np.uint8)

# 8. Display and save results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image_rgb)
axes[0].set_title("Original Image")

axes[1].imshow(best_mask, cmap="gray")
axes[1].set_title("Selected SAM Mask")

axes[2].imshow(overlay)
axes[2].set_title("Egg Mask Overlay")

for axis in axes:
    axis.axis("off")

plt.tight_layout()

result_path = os.path.join(OUTPUT_DIR, "result.png")
plt.savefig(result_path, dpi=150)
plt.show()

print("Saved mask:", mask_path)
print("Saved visualization:", result_path)
