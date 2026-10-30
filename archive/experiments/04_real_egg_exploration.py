import cv2
import matplotlib.pyplot as plt
import glob
import os

# Find images inside the dataset.
image_paths = glob.glob(
    "dataset/valid/egg/*.jpg"
)

# Check whether images were found.
if not image_paths:
    raise FileNotFoundError(
        "No images found in dataset/valid/egg/"
    )

# Select the first image.
image_path = image_paths[0]

print("Selected image:", image_path)

# Load the image.
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(
        f"Could not read image: {image_path}"
    )

# Convert BGR to RGB for displaying colors correctly.
image_rgb = cv2.cvtColor(
    image, cv2.COLOR_BGR2RGB
)

# Convert the image to grayscale.
gray = cv2.cvtColor(
    image, cv2.COLOR_BGR2GRAY
)

# Display both images.
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Egg Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.tight_layout()
plt.show()

# Print image details.
print("Image loaded successfully!")
print("Image dimensions:", image.shape)
print("Grayscale dimensions:", gray.shape)
