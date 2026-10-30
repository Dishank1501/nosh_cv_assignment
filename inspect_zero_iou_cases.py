import os
import cv2
import pandas as pd
import numpy as np

CSV = "outputs/trial5_grabcut_fallback/metrics.csv"
OUT = "outputs/zero_iou_first10.jpg"
SIZE = 260

df = pd.read_csv(CSV)
failures = df[df["iou_vs_sam"] == 0].head(10)

if failures.empty:
    print("No zero-IoU cases found.")
    raise SystemExit

def find_reference(image_path):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    prefix = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
    candidate = f"{prefix}__{stem}_mask.png"

    path = os.path.join("outputs/automatic_masks", candidate)
    if os.path.exists(path):
        return path

    # Fallback: search by unique image stem.
    for name in os.listdir("outputs/automatic_masks"):
        if name.endswith("_mask.png") and stem in name:
            return os.path.join("outputs/automatic_masks", name)

    return None

def load_panel(path, is_mask=False):
    if not path or not os.path.exists(path):
        img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        cv2.putText(img, "NOT FOUND", (20, SIZE // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return img

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((SIZE, SIZE, 3), dtype=np.uint8)

    if is_mask:
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        out = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        out[img > 0] = (0, 255, 0)
        return out

    h, w = img.shape[:2]
    scale = min(SIZE / w, SIZE / h)
    nw, nh = int(w * scale), int(h * scale)
    img = cv2.resize(img, (nw, nh))
    canvas = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    x, y = (SIZE - nw) // 2, (SIZE - nh) // 2
    canvas[y:y+nh, x:x+nw] = img
    return canvas

headers = ["Original", "SAM reference", "Trial 5", "Trial 8"]
row_height = SIZE + 45
sheet = np.zeros((len(failures) * row_height, 4 * SIZE, 3), dtype=np.uint8)

for i, (_, row) in enumerate(failures.iterrows()):
    image_path = row["image"]
    ref_path = find_reference(image_path)
    trial5_path = row["prediction_mask"]

    stem = os.path.splitext(os.path.basename(image_path))[0]
    trial8_path = f"outputs/trial8_conservative_refinement/masks/{stem}_mask.png"

    paths = [image_path, ref_path, trial5_path, trial8_path]

    for j, (header, path) in enumerate(zip(headers, paths)):
        x = j * SIZE
        y = i * row_height

        cv2.putText(sheet, header, (x + 5, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

        panel = load_panel(path, is_mask=(j in [1, 2, 3]))
        sheet[y + 35:y + 35 + SIZE, x:x + SIZE] = panel

    cv2.putText(
        sheet,
        f"{stem[:42]} | IoU=0",
        (5, i * row_height + SIZE + 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
        (255, 255, 255), 1, cv2.LINE_AA
    )

os.makedirs("outputs", exist_ok=True)
cv2.imwrite(OUT, sheet)
print(f"Saved: {OUT}")
print(f"Reviewed {len(failures)} zero-IoU cases.")
