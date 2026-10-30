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

# Load image
image_bgr = cv2.imread(IMAGE_PATH)

if image_bgr is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Load SAM
torch.set_num_threads(4)

print("Loading SAM...")
sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
sam.to(device="cpu")
sam.eval()

predictor = SamPredictor(sam)

print("Preparing image for SAM...")
predictor.set_image(image_rgb)

# Click inside the egg to select it
clicked_points = []

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(image_rgb)
ax.set_title("Click once inside the egg")
ax.axis("off")


def on_click(event):
    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return

    x = int(event.xdata)
    y = int(event.ydata)

    clicked_points.append([x, y])

    ax.scatter(x, y, c="lime", s=100, marker="x")
    fig.canvas.draw_idle()

    print(f"Selected point: x={x}, y={y}")

    if len(clicked_points) == 1:
        plt.close(fig)


fig.canvas.mpl_connect("button_press_event", on_click)
plt.show()

if not clicked_points:
    raise RuntimeError("No point selected. Run the script and click inside the egg.")

point_coords = np.array(clicked_points, dtype=np.float32)
point_labels = np.ones(len(clicked_points), dtype=np.int32)

# Predict masks using the selected point
print("Generating point-prompted masks...")

with torch.inference_mode():
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

# Choose the candidate with the highest predicted score
best_index = int(np.argmax(scores))
best_mask = masks[best_index]

print("Candidate scores:", scores)
print("Selected candidate:", best_index)
print("Predicted quality score:", round(float(scores[best_index]), 4))
print("Mask area:", int(np.count_nonzero(best_mask)))

# Save binary mask
import os

os.makedirs("outputs/point_prompt", exist_ok=True)

cv2.imwrite(
    "outputs/point_prompt/egg_mask.png",
    best_mask.astype(np.uint8) * 255,
)

# Create overlay
overlay = image_rgb.copy()
overlay[best_mask] = (
    0.5 * overlay[best_mask]
    + 0.5 * np.array([255, 0, 0])
).astype(np.uint8)

# Display results
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
plt.savefig("outputs/point_prompt/result.png", dpi=150)
plt.show()

print("Saved mask and visualization in outputs/point_prompt/")
