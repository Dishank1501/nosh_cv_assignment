import glob
import os
import cv2
import matplotlib.pyplot as plt

overlay_paths = sorted(
    glob.glob("outputs/batch_test/*_overlay.jpg")
)

if not overlay_paths:
    raise FileNotFoundError("No overlay images found.")

plt.figure(figsize=(15, 10))

for index, path in enumerate(overlay_paths):
    image_bgr = cv2.imread(path)

    if image_bgr is None:
        print("Could not read:", path)
        continue

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    plt.subplot(2, 3, index + 1)
    plt.imshow(image_rgb)
    plt.title(os.path.basename(path))
    plt.axis("off")

plt.tight_layout()
plt.show()
