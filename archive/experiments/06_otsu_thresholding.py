import cv2
import matplotlib.pyplot as plt
import glob

# Find an image from the dataset
image_paths = glob.glob("dataset/valid/egg/*.jpg")

if not image_paths:
    raise FileNotFoundError("No egg images found.")

image_path = image_paths[0]
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not read {image_path}")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Reduce noise using Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Automatically calculate the threshold using Otsu's method
threshold_value, binary_mask = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

print("Image:", image_path)
print("Otsu threshold selected:", threshold_value)

# Display the results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(blurred, cmap="gray")
plt.title("Gaussian Blur")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(binary_mask, cmap="gray")
plt.title("Otsu Binary Mask")
plt.axis("off")

plt.tight_layout()
plt.show()

