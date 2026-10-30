import cv2
import matplotlib.pyplot as plt
import glob

# Find an egg image from the dataset
image_paths = glob.glob("dataset/valid/egg/*.jpg")

if not image_paths:
    raise FileNotFoundError(
        "No images found in dataset/valid/egg/"
    )

image_path = image_paths[0]
print("Using image:", image_path)

# Load the image
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not read: {image_path}")

# Convert BGR to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Try three threshold values
threshold_values = [100, 150, 200]

plt.figure(figsize=(15, 5))

for index, threshold_value in enumerate(threshold_values):
    _, binary_mask = cv2.threshold(
        gray,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    plt.subplot(1, 3, index + 1)
    plt.imshow(binary_mask, cmap="gray")
    plt.title(f"Threshold = {threshold_value}")
    plt.axis("off")

plt.tight_layout()
plt.show()
