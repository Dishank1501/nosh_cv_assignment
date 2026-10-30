import os
import glob
import csv

import cv2
import torch
import numpy as np

from segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator,
)


CHECKPOINT = "models/sam_vit_b_01ec64.pth"
OUTPUT_DIR = "outputs/automatic_masks"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Find all dataset images
image_paths = sorted(
    glob.glob("dataset/train/**/*.jpg", recursive=True)
    + glob.glob("dataset/valid/**/*.jpg", recursive=True)
    + glob.glob("dataset/test/**/*.jpg", recursive=True)
)

if not image_paths:
    raise FileNotFoundError("No dataset images found.")

print("Total images:", len(image_paths))

torch.set_num_threads(4)


# Load SAM once
print("Loading SAM ViT-B...")
sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
sam.to(device="cpu")
sam.eval()

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=8,
    points_per_batch=4,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.90,
    crop_n_layers=0,
    min_mask_region_area=100,
)


# Save a record of the results
csv_path = os.path.join(OUTPUT_DIR, "results.csv")

with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)

    writer.writerow([
        "image_path",
        "candidate_count",
        "selected_candidate",
        "mask_area",
        "predicted_iou",
        "stability_score",
        "custom_score",
        "status",
    ])

    for index, image_path in enumerate(image_paths, start=1):
        print(f"\n[{index}/{len(image_paths)}] {image_path}")

        image_bgr = cv2.imread(image_path)

        if image_bgr is None:
            print("ERROR: Could not read image.")
            writer.writerow([
                image_path, 0, -1, 0, 0, 0, 0, "read_error"
            ])
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        height, width = image_rgb.shape[:2]
        image_area = height * width

        # Generate candidate masks
        try:
            with torch.inference_mode():
                masks = mask_generator.generate(image_rgb)
        except Exception as error:
            print("ERROR during SAM:", error)
            writer.writerow([
                image_path, 0, -1, 0, 0, 0, 0, "sam_error"
            ])
            continue

        if not masks:
            print("No candidate masks generated.")
            writer.writerow([
                image_path, 0, -1, 0, 0, 0, 0, "no_masks"
            ])
            continue

        image_cx = width / 2
        image_cy = height / 2

        best_mask = None
        best_score = float("-inf")
        best_index = -1

        # Rank candidates using geometry and SAM scores
        for candidate_index, candidate in enumerate(masks):
            segmentation = candidate["segmentation"]
            area = int(np.count_nonzero(segmentation))

            if area == 0:
                continue

            area_ratio = area / image_area

            # Initial experimental area filter
            if area_ratio < 0.02 or area_ratio > 0.40:
                continue

            x, y, box_width, box_height = candidate["bbox"]

            box_cx = x + box_width / 2
            box_cy = y + box_height / 2

            center_distance = np.sqrt(
                ((box_cx - image_cx) / width) ** 2
                + ((box_cy - image_cy) / height) ** 2
            )

            center_score = max(0.0, 1.0 - center_distance)

            aspect_ratio = box_width / max(box_height, 1)

            shape_score = (
                1.0 if 1.1 <= aspect_ratio <= 2.2 else 0.0
            )

            predicted_iou = float(candidate["predicted_iou"])
            stability_score = float(candidate["stability_score"])

            # Custom ranking score (not a probability or measured IoU)
            score = (
                0.40 * predicted_iou
                + 0.25 * stability_score
                + 0.25 * center_score
                + 0.10 * shape_score
            )

            if score > best_score:
                best_score = score
                best_mask = segmentation
                best_index = candidate_index

        # Save the selected mask
        relative_path = os.path.relpath(image_path, "dataset")
        safe_name = relative_path.replace(os.sep, "__")
        base_name = os.path.splitext(safe_name)[0]

        if best_mask is None:
            print("No candidate passed the filters.")
            writer.writerow([
                image_path,
                len(masks),
                -1,
                0,
                0,
                0,
                0,
                "no_candidate",
            ])
            continue

        mask_path = os.path.join(
            OUTPUT_DIR, base_name + "_mask.png"
        )

        overlay_path = os.path.join(
            OUTPUT_DIR, base_name + "_overlay.jpg"
        )

        cv2.imwrite(
            mask_path,
            best_mask.astype(np.uint8) * 255,
        )

        overlay = image_rgb.copy()
        foreground = best_mask.astype(bool)

        overlay[foreground] = (
            0.5 * overlay[foreground].astype(np.float32)
            + 0.5 * np.array([255, 0, 0], dtype=np.float32)
        ).astype(np.uint8)

        cv2.imwrite(
            overlay_path,
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )

        selected = masks[best_index]

        writer.writerow([
            image_path,
            len(masks),
            best_index,
            int(np.count_nonzero(best_mask)),
            float(selected["predicted_iou"]),
            float(selected["stability_score"]),
            best_score,
            "selected",
        ])

        print("Candidates:", len(masks))
        print("Selected candidate:", best_index)
        print("Custom selection score:", best_score)
        print("Mask area:", int(np.count_nonzero(best_mask)))
        print("Saved mask:", mask_path)
        print("Saved overlay:", overlay_path)

print("\nBatch processing completed.")
print("Results CSV:", csv_path)