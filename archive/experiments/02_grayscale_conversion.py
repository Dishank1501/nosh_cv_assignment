import cv2
import matplotlib.pyplot as plt

# Load the image we created earlier
image = cv2.imread("first_image.png")

# Check whether the image loaded
if image is None:
    raise FileNotFoundError("first_image.png not found")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display both images
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.tight_layout()
plt.show()

# Print image dimensions
print("Original shape:", image.shape)
print("Grayscale shape:", gray.shape)

# Save the grayscale image
cv2.imwrite("grayscale.png", gray)