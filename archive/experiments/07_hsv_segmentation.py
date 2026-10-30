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

# Convert BGR to RGB for display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert BGR to HSV for color analysis
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Split HSV into three channels
hue, saturation, value = cv2.split(hsv)

# Create a mask for low-saturation, bright regions
mask = cv2.inRange(
    hsv,
    (0, 0, 170),
    (179, 100, 255)
)

# Display results
plt.figure(figsize=(15, 8))

plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(saturation, cmap="gray")
plt.title("Saturation Channel")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(value, cmap="gray")
plt.title("Brightness Channel")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(mask, cmap="gray")
plt.title("Low-Saturation, Bright Regions")
plt.axis("off")

plt.tight_layout()
plt.show()

