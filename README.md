# Nosh Robotics Egg Segmentation

## Setup

The verified workflow environment uses Python 3.10.12, OpenCV 5.0.0, and NumPy 2.2.6.

Create and configure the virtual environment:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install numpy opencv-python pandas
```

The E0–E5 workflow requires OpenCV and NumPy. Historical trial scripts also use pandas.

SAM masks and the ViT-B checkpoint are stored locally under `outputs/automatic_masks/` and `models/`. Regenerating these assets is not part of the normal workflow. The optional `batch_sam_segmentation.py` script writes to the reference directory and overwrites `results.csv`. Running SAM generation additionally requires PyTorch and `segment_anything`.

## Data and Results

The dataset contains 110 images under `dataset/`.

The custom manifests are:

- `splits/dev.txt` — 88 images
- `splits/eval.txt` — 22 images

The manifests are disjoint and collectively cover the dataset. However, their membership crosses the dataset's original `train/`, `valid/`, and `test/` folders. In particular, `dev.txt` includes nine images from the folder named `test`.

All reported segmentation metrics measure agreement with SAM-generated pseudo-labels. They should not be interpreted as accuracy against independently verified ground-truth annotations.

The additive E0–E5 workflow is implemented in:

```text
src/egg_workflow.py
```

Results are stored locally under `outputs/e0_*` through `outputs/e5_*`. Historical Trial 5–10 results remain in their existing output directories.

The experiment log, `experiment_log.md`, documents the methods, results, integrity checks, and submission limitations.

## Run and Verify

From the project root:

```bash
cd /home/dishank/nosh_cv_assignment

# Run unit tests
.venv/bin/python -m unittest discover -s tests -v

# Audit dataset and workflow inputs
.venv/bin/python src/egg_workflow.py audit

# Run workflow stages
.venv/bin/python src/egg_workflow.py e0
.venv/bin/python src/egg_workflow.py e1
.venv/bin/python src/egg_workflow.py e2
.venv/bin/python src/egg_workflow.py e4
.venv/bin/python src/egg_workflow.py e5
```

Run E5 only after selecting the segmentation method using development data. The workflow refuses to overwrite an existing `outputs/e5_final_evaluation/` directory.

The existing E5 result in this workspace uses Trial 5. Evaluation artifacts were already present, so this result should not be described as a pristine blind-test evaluation.

## Tracking

A tracking implementation is available at:

```text
src/track_eggs_kalman_hungarian.py
```

Generated tracking artifacts are stored under `outputs/tracking/`. These output files are excluded from the GitHub submission. Tracking identity continuity should be evaluated separately before making claims about end-to-end tracking performance.

## Submission Notes

The GitHub repository is intended to contain source code, tests, documentation, and configuration files.

Large or generated assets are excluded, including:

- Dataset images
- Generated segmentation outputs
- SAM model checkpoint
- Generated tracking videos and CSV files

The local workspace retains these assets for reproducibility and further evaluation.