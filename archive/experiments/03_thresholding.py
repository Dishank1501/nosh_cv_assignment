import cv2
import matplotlib.pyplot as plt

# Load the grayscale image.
gray = cv2.imread("grayscale.png", cv2.IMREAD_GRAYSCALE)

if gray is None:
    raise FileNotFoundError("grayscale.png not found")

# Set the threshold value.
threshold_value = 180

# Convert the grayscale image into a binary image.
_, binary = cv2.threshold(
    gray,
    threshold_value,
    255,
    cv2.THRESH_BINARY
)

# Display both images.
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(binary, cmap="gray")
plt.title("Binary Image")
plt.axis("off")

plt.tight_layout()
plt.show()

# Save the binary mask.
cv2.imwrite("binary_mask.png", binary)

print("Threshold value:", threshold_value)
print("Binary mask saved successfully.")
