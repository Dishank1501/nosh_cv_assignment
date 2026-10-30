from pathlib import Path

import cv2
import numpy as np

image_paths = [
    "dataset/train/egg/not_damaged_118_jpg.rf.3e17ba84bcb3b57ebbf743b145c5ae27.jpg",
    "dataset/valid/egg/not_damaged_126_jpg.rf.2c3342cf9ca7ab55f6682ef198e98bbd.jpg",
    "dataset/test/egg/not_damaged_153_jpg.rf.ea842e70f08e63e69bbcc74f1bc143a8.jpg",
    "dataset/train/Unlabeled/not_damaged_135_jpg.rf.a6d81ffb82470e031fb3ab469c8f0db2.jpg",
    "dataset/train/egg/not_damaged_121_jpg.rf.0506fe180f5fed88ae65348130fb5192.jpg",
]

out_dir = Path("outputs/evaluation_failure_review")
out_dir.mkdir(parents=True, exist_ok=True)

cell_w, cell_h = 320, 250
sheet = np.full((len(image_paths) * cell_h, cell_w * 3, 3), 255, dtype=np.uint8)
valid_count = 0

for i, image_path in enumerate(image_paths):
    relative = Path(image_path).relative_to("dataset")
    filename = str(relative).replace("/", "__")
    filename = Path(filename).with_suffix("").name

    image = cv2.imread(image_path)
    sam = cv2.imread(
        f"outputs/automatic_masks/{filename}_mask.png",
        cv2.IMREAD_GRAYSCALE,
    )
    grabcut = cv2.imread(
        f"outputs/grabcut_eval/{filename}_mask.png",
        cv2.IMREAD_GRAYSCALE,
    )

    if any(x is None for x in (image, sam, grabcut)):
        print("Missing file:", image_path)
        continue

    scale = min(310 / image.shape[1], 190 / image.shape[0])
    size = (int(image.shape[1] * scale), int(image.shape[0] * scale))

    panels = [
        ("Original", cv2.resize(image, size)),
        ("SAM reference", cv2.cvtColor(
            cv2.resize(sam, size, interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_GRAY2BGR,
        )),
        ("GrabCut prediction", cv2.cvtColor(
            cv2.resize(grabcut, size, interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_GRAY2BGR,
        )),
    ]

    y = valid_count * cell_h
    cv2.putText(
        sheet, Path(image_path).name[:42], (5, y + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1,
    )

    for j, (title, panel) in enumerate(panels):
        x = j * cell_w
        h, w = panel.shape[:2]
        sheet[y + 30:y + 30 + h, x:x + w] = panel
        cv2.putText(
            sheet, title, (x + 5, y + 225),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
        )

    valid_count += 1

sheet = sheet[:valid_count * cell_h]
output = out_dir / "five_worst_examples.jpg"
cv2.imwrite(str(output), sheet)

print("Examples included:", valid_count)
print("Saved:", output)
