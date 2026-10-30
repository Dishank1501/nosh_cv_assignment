import cv2
import numpy as np

# Create a black image:
# 300 pixels high, 400 pixels wide, 3 color channels.
image = np.zeros((300, 400, 3), dtype=np.uint8)

# Draw a white circle.
cv2.circle(image, (200, 150), 70, (255, 255, 255), -1)

# Draw a red rectangle.
# OpenCV uses BGR color order.
cv2.rectangle(image, (20, 20), (100, 80), (0, 0, 255), -1)

# Save the image.
cv2.imwrite("first_image.png", image)

# Print image information.
print("Image shape:", image.shape)
print("Data type:", image.dtype)
print("Center pixel:", image[150, 200])

print("Image saved successfully!")