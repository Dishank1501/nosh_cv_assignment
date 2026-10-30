from pathlib import Path

import cv2
import numpy as np


def segment_egg(image_bgr):
    height, width = image_bgr.shape[:2]

    # Automatically initialize GrabCut with a central rectangle.
    margin_x = int(width * 0.10)
    margin_y = int(height * 0.10)

    rect = (
        margin_x,
        margin_y,
        width - 2 * margin_x,
        height - 2 * margin_y,
    )

    mask = np.zeros((height, width), dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(
            image_bgr,
            mask,
            rect,
            background_model,
            foreground_model,
            5,
            cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error as error:
        print("GrabCut failed:", error)
        return np.zeros((height, width), dtype=np.uint8)

    # Definite and probable foreground pixels.
    result = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    # Remove small regions and fill small gaps.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5)
    )
    result = cv2.morphologyEx(
        result, cv2.MORPH_OPEN, kernel
    )
    result = cv2.morphologyEx(
        result, cv2.MORPH_CLOSE, kernel
    )

    # Keep the largest plausible connected component.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        result, connectivity=8
    )

    cleaned = np.zeros_like(result)
    image_area = height * width
    candidates = []

    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        ratio = area / image_area

        if 0.02 <= ratio <= 0.40:
            candidates.append((area, label))

    if candidates:
        _, best_label = max(candidates)
        cleaned[labels == best_label] = 255

    return cleaned


def main():
    paths = [
        line.strip()
        for line in Path("splits/dev.txt").read_text().splitlines()
        if line.strip()
    ]

    output_dir = Path("outputs/grabcut_dev")
    output_dir.mkdir(parents=True, exist_ok=True)

    successful = 0

    for index, image_path in enumerate(paths, start=1):
        image = cv2.imread(image_path)

        if image is None:
            print(f"[{index}/{len(paths)}] Could not read: {image_path}")
            continue

        mask = segment_egg(image)

        relative = Path(image_path).relative_to("dataset")
        filename = str(relative).replace("/", "__")
        filename = Path(filename).with_suffix("").name

        cv2.imwrite(
            str(output_dir / f"{filename}_mask.png"),
            mask,
        )

        successful += 1
        print(f"[{index}/{len(paths)}] Processed")

    print(f"\nProcessed successfully: {successful}/{len(paths)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
