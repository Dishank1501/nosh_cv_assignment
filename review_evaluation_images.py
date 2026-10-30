import os
import cv2
import numpy as np

images = []

with open("splits/eval.txt") as f:
    paths = [line.strip() for line in f if line.strip()]

cell_w, cell_h = 220, 190
cols = 4
rows = (len(paths) + cols - 1) // cols

sheet = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

for i, path in enumerate(paths):
    img = cv2.imread(path)
    if img is None:
        print("Could not read:", path)
        continue

    h, w = img.shape[:2]
    scale = min((cell_w - 10) / w, (cell_h - 40) / h)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))

    x = (i % cols) * cell_w
    y = (i // cols) * cell_h

    sheet[y:y + img.shape[0], x:x + img.shape[1]] = img

    label = os.path.basename(path)[:28]
    cv2.putText(
        sheet, label, (x + 3, y + cell_h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
        (255, 255, 255), 1, cv2.LINE_AA
    )

os.makedirs("outputs", exist_ok=True)
cv2.imwrite("outputs/eval_image_review.jpg", sheet)
print("Saved outputs/eval_image_review.jpg")
print("Evaluation images:", len(paths))
