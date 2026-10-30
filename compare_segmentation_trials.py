from pathlib import Path
import csv
import cv2
import numpy as np

ROOT = Path.home() / "nosh_cv_assignment"

FILES = {
    "trial5": ROOT / "outputs/trial5_grabcut_fallback/metrics.csv",
    "trial7": ROOT / "outputs/trial7_lab_guided_grabcut/metrics.csv",
}

OUTPUT = ROOT / "outputs/trial5_vs_trial7_comparison.jpg"

SELECTED = [
    "not_damaged_164_jpg.rf.aeab74f44b77d69cb4884f8a55d4a5b8",
    "not_damaged_148_jpg.rf.8a3e1c0e1dae236abe0e9a75a4882bda",
    "not_damaged_140_jpg.rf.5a28b6a69e229bf384854ec1e98b3f19",
    "not_damaged_156_jpg.rf.6ae4742599d61d6120e5beb5d38bcf2e",
]

def load_csv(path, iou_column):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        Path(r["image"]).stem: {
            "row": r,
            "iou": float(r[iou_column]),
        }
        for r in rows
    }

t5 = load_csv(FILES["trial5"], "iou_vs_sam")
t7 = load_csv(FILES["trial7"], "iou_vs_sam_pseudolabel")

def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image

def color_mask(mask):
    rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    rgb[mask > 0] = (0, 220, 0)
    return rgb

def add_title(image, title):
    image = image.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 32), (35, 35, 35), -1)
    cv2.putText(
        image, title, (7, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1,
        cv2.LINE_AA
    )
    return image

tiles = []
tile_width = 320
tile_height = 320

for stem in SELECTED:
    if stem not in t5 or stem not in t7:
        print("Missing CSV row:", stem)
        continue

    r5 = t5[stem]["row"]
    r7 = t7[stem]["row"]

    # Locate original image from the dataset path recorded in the CSV.
    original_path = ROOT / r5["image"]
    original = cv2.imread(str(original_path))
    if original is None:
        print("Cannot read original:", original_path)
        continue

    mask5_path = ROOT / r5["prediction_mask"]
    mask7_path = ROOT / r7["mask_path"]

    mask5 = read_gray(mask5_path)
    mask7 = read_gray(mask7_path)

    # Reference masks are indexed by the image stem in the original output.
    reference_path = None
    for p in (ROOT / "outputs/automatic_masks").glob("*_mask.png"):
        if p.name[:-len("_mask.png")].rsplit("__", 1)[-1] == stem:
            reference_path = p
            break

    if reference_path is None:
        print("SAM reference missing:", stem)
        continue

    reference = read_gray(reference_path)

    panels = [
        add_title(original, "Original"),
        add_title(color_mask(reference), "SAM reference"),
        add_title(color_mask(mask5), f"Trial 5 IoU={t5[stem]['iou']:.3f}"),
        add_title(color_mask(mask7), f"Trial 7 IoU={t7[stem]['iou']:.3f}"),
    ]

    row_panels = []
    for panel in panels:
        panel = cv2.resize(panel, (tile_width, tile_height))
        row_panels.append(panel)

    combined = np.hstack(row_panels)
    cv2.putText(
        combined, stem[:65], (8, tile_height - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1,
        cv2.LINE_AA
    )
    tiles.append(combined)

if not tiles:
    raise SystemExit("No comparison images were generated.")

sheet = np.vstack(tiles)
cv2.imwrite(str(OUTPUT), sheet)
print("Saved comparison sheet:", OUTPUT)
