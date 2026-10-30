from pathlib import Path

import cv2
import numpy as np


def segment_egg(image_bgr):
    """
    Classical baseline:
    1. Convert the image to HSV.
    2. Threshold bright, low-saturation pixels.
    3. Clean the binary mask using morphological operations.
    4. Keep the largest plausible connected component.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Initial parameters; these will be evaluated on the dev set.
    lower = np.array([0, 0, 140], dtype=np.uint8)
    upper = np.array([179, 120, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (7, 7)
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    cleaned = np.zeros(mask.shape, dtype=np.uint8)
    image_area = mask.shape[0] * mask.shape[1]

    candidates = []
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        area_ratio = area / image_area

        if 0.02 <= area_ratio <= 0.40:
            candidates.append((area, label))

    if candidates:
        _, best_label = max(candidates)
        cleaned[labels == best_label] = 255

    return cleaned


def main():
    split_file = Path("splits/dev.txt")
    output_dir = Path("outputs/classical_dev")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        line.strip()
        for line in split_file.read_text().splitlines()
        if line.strip()
    ]

    successful = 0

    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(image_path)

        if image is None:
            print(f"[{index}/{len(image_paths)}] Could not read: {image_path}")
            continue

        mask = segment_egg(image)

        relative_path = Path(image_path).relative_to("dataset")
        filename = str(relative_path).replace("/", "__")
        filename = Path(filename).with_suffix("").name

        mask_path = output_dir / f"{filename}_mask.png"
        cv2.imwrite(str(mask_path), mask)
        successful += 1

        print(f"[{index}/{len(image_paths)}] Saved {mask_path}")

    print(f"\nProcessed successfully: {successful}/{len(image_paths)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
