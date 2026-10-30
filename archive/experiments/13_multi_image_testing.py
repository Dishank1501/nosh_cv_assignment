import os
import glob
import cv2
import torch
import numpy as np

from segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator,
)

CHECKPOINT = "models/sam_vit_b_01ec64.pth"
OUTPUT_DIR = "outputs/batch_test"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Start with five images from the validation folder
image_paths = sorted(
    glob.glob("dataset/valid/egg/*.jpg")
)[:5]

if not image_paths:
    raise FileNotFoundError("No validation images found.")

torch.set_num_threads(4)

# Load SAM only once
print("Loading SAM...")
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

for image_index, image_path in enumerate(image_paths, start=1):
    print(f"\nProcessing image {image_index}/{len(image_paths)}")
    print("File:", os.path.basename(image_path))

    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        print("Could not read image; skipping.")
        continue

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    image_area = height * width

    # Generate candidate masks
    with torch.inference_mode():
        masks = mask_generator.generate(image_rgb)

    if not masks:
        print("No candidate masks generated.")
        continue

    image_cx = width / 2
    image_cy = height / 2

    best_mask = None
    best_score = float("-inf")
    best_index = -1

    # Rank candidates using geometric heuristics
    for candidate_index, candidate in enumerate(masks):
        segmentation = candidate["segmentation"]
        area = int(np.count_nonzero(segmentation))

        if area == 0:
            continue

        x, y, box_width, box_height = candidate["bbox"]
        area_ratio = area / image_area

        # Ignore tiny masks and very large regions
        if area_ratio < 0.02 or area_ratio > 0.40:
            continue

        box_cx = x + box_width / 2
        box_cy = y + box_height / 2

        center_distance = np.sqrt(
            ((box_cx - image_cx) / width) ** 2
            + ((box_cy - image_cy) / height) ** 2
        )

        # SAM's estimated mask-quality scores
        predicted_iou = float(candidate["predicted_iou"])
        stability_score = float(candidate["stability_score"])

        # Geometric suitability
        center_score = max(0.0, 1.0 - center_distance)

        aspect_ratio = box_width / max(box_height, 1)

        if 1.1 <= aspect_ratio <= 2.2:
            shape_score = 1.0
        else:
            shape_score = 0.0

        # Combined heuristic score
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

    if best_mask is None:
        print("No candidate passed the geometric filters.")
        continue

    # Save selected mask
    mask_output = (
        f"{OUTPUT_DIR}/image_{image_index:02d}_mask.png"
    )

    cv2.imwrite(
        mask_output,
        best_mask.astype(np.uint8) * 255,
    )

    # Create an overlay for visual inspection
    overlay = image_rgb.copy()
    foreground = best_mask.astype(bool)

    overlay[foreground] = (
        0.5 * overlay[foreground]
        + 0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)

    overlay_output = (
        f"{OUTPUT_DIR}/image_{image_index:02d}_overlay.jpg"
    )

    cv2.imwrite(
        overlay_output,
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )

    print("Candidate selected:", best_index)
    print("Mask area:", int(np.count_nonzero(best_mask)))
    selected_candidate = masks[best_index]

    print("Selection score:", round(best_score, 3))
    print(
        "Predicted IoU:",
        round(float(selected_candidate["predicted_iou"]), 3)
    )
    print(
        "Stability score:",
        round(float(selected_candidate["stability_score"]), 3)
    )
    print("Saved mask:", mask_output)
    print("Saved overlay:", overlay_output)

    print("\nBatch testing completed.")
    print("Inspect the saved overlays in:", OUTPUT_DIR)

