# Nosh Robotics Egg Segmentation — Experiment Log

## 1. Project Overview

This project explores classical computer vision techniques for egg segmentation, evaluates multiple segmentation approaches, and develops an additive E0–E5 workflow.

The experiments compare segmentation outputs against SAM-generated pseudo-labels. These pseudo-labels provide a consistent reference for experimentation but are not independently verified ground-truth annotations.

The project also includes a separate egg-tracking implementation using Kalman filtering and Hungarian assignment.

---

## 2. Environment and Dependencies

Verified environment:

- Python: 3.10.12
- OpenCV: 5.0.0
- NumPy: 2.2.6

Install the main dependencies:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install numpy opencv-python pandas
```

The E0–E5 workflow requires OpenCV and NumPy. Historical experiment scripts also use pandas.

Optional SAM generation dependencies:

- PyTorch
- segment_anything

SAM masks and the ViT-B checkpoint are retained as local reference assets. Regenerating them is not required for the normal workflow.

---

## 3. Dataset and Evaluation Setup

The dataset contains 110 images.

Custom split manifests:

| Split | Images |
|---|---:|
| Development | 88 |
| Evaluation | 22 |
| Total | 110 |

Manifest files:

- `splits/dev.txt`
- `splits/eval.txt`

The manifests are disjoint and collectively cover all 110 images.

### Dataset integrity observations

The custom split membership crosses the dataset's original `train/`, `valid/`, and `test/` folders. In particular, the development manifest includes nine images from the folder named `test`.

Therefore, the custom development/evaluation split should be treated as the actual split used for these experiments rather than assuming the original folder names represent the experiment split.

### Evaluation limitation

All reported segmentation metrics measure agreement with SAM pseudo-labels.

They do not establish accuracy against independently annotated ground truth. Consequently, results should be described as pseudo-label agreement rather than verified segmentation accuracy.

---

## 4. Historical Segmentation Experiments

### Trial 5 — Classical Segmentation Baseline

Trial 5 was selected as the baseline for subsequent comparisons.

The approach uses:

1. LAB colour-space conversion.
2. Illumination correction using the L channel and Gaussian background estimation with sigma 80.
3. Normalization of the corrected image.
4. Median filtering with a 5 × 5 kernel.
5. Otsu thresholding using both threshold polarities.
6. Morphological processing with a 5 × 5 kernel.
7. Contour-based candidate filtering.

Candidate filtering uses the following constraints:

- Area between 2% and 40% of the image.
- Aspect ratio no greater than 2.8.
- Circularity of at least 0.25.
- Largest qualifying contour selection.

A central elliptical GrabCut fallback is used when the threshold-based approach does not produce a suitable candidate.

#### Trial 5 results

| Metric | Development (88 images) | Evaluation (22 images) |
|---|---:|---:|
| Mean IoU | 0.4029 | 0.3576 |
| Median IoU | 0.4069 | 0.2469 |
| Mean Dice | 0.4650 | 0.4253 |
| Median Dice | 0.5784 | 0.3958 |
| Zero-IoU cases | 30 | 8 |

Trial 5 remained the selected baseline after comparison with later experiments.

The evaluation artifacts were already present in the workspace. Therefore, the evaluation result is not a pristine blind-test result.

---

### Trial 6 — Ellipse Refinement

Trial 6 explored ellipse-based refinement of the segmentation output.

| Metric | Development |
|---|---:|
| Mean IoU | 0.4023 |
| Mean Dice | 0.4663 |
| Zero-IoU cases | 31 |

The results did not improve on Trial 5.

---

### Trial 7 — LAB-Guided GrabCut

Trial 7 explored LAB-guided GrabCut segmentation.

| Metric | Development |
|---|---:|
| Mean IoU | 0.3024 |
| Mean Dice | 0.3404 |
| Zero-IoU cases | 45 |

The approach performed worse than Trial 5.

---

### Trial 8 — Conservative Refinement

Trial 8 explored conservative candidate refinement.

| Metric | Development |
|---|---:|
| Mean IoU | 0.4033 |
| Mean Dice | 0.4653 |
| Zero-IoU cases | 30 |

The improvement over Trial 5 was negligible and did not justify replacing the baseline.

---

### Trial 9 — Multiple Candidate Segmentation

Trial 9 explored multiple candidate generation and selection.

| Metric | Development |
|---|---:|
| Mean IoU | 0.2117 |
| Mean Dice | 0.2956 |
| Zero-IoU cases | 14 |

The overall metric performance was substantially lower than Trial 5.

---

### Trial 10 — Conservative Candidate Generation

Trial 10 explored a more conservative candidate-generation strategy.

| Metric | Development |
|---|---:|
| Mean IoU | 0.2375 |
| Mean Dice | 0.3112 |
| Zero-IoU cases | 31 |

The results did not outperform Trial 5.

---

## 5. Historical Experiment Comparison

| Trial | Mean IoU | Mean Dice | Zero-IoU cases |
|---|---:|---:|---:|
| Trial 5 | 0.4029 | 0.4650 | 30 |
| Trial 6 | 0.4023 | 0.4663 | 31 |
| Trial 7 | 0.3024 | 0.3404 | 45 |
| Trial 8 | 0.4033 | 0.4653 | 30 |
| Trial 9 | 0.2117 | 0.2956 | 14 |
| Trial 10 | 0.2375 | 0.3112 | 31 |

Trial 5 was retained as the selected baseline.

Trial 8 produced a marginally higher mean IoU, but the difference was negligible. It did not provide a meaningful overall improvement sufficient to replace Trial 5.

The zero-IoU count should be interpreted alongside the mean and median metrics; it does not independently establish overall segmentation quality.

---

## 6. Additive E0–E5 Workflow

The additive workflow is implemented in:

`src/egg_workflow.py`

The workflow is organized into stages E0 through E5. The stages provide a structured way to audit inputs, execute the segmentation workflow, and evaluate the selected approach.

### Reproduction and integrity checks

The E0–E5 implementation was checked against the historical Trial 5 development results.

The reproduced development results matched the original Trial 5 results across all 88 images:

- Images reproduced: 88/88
- Maximum IoU difference: approximately 0.0000005

The small difference is attributable to CSV metric rounding.

This provides a reproducibility check for the implemented workflow.

### Existing output locations

Workflow outputs are stored locally under:

- `outputs/e0_*`
- `outputs/e1_*`
- `outputs/e2_*`
- `outputs/e3_*`
- `outputs/e4_*`
- `outputs/e5_*`

Historical Trial 5–10 outputs remain in their existing directories.

### Evaluation safeguards

E5 is intended to be run after selecting the method using development data.

The workflow refuses to overwrite an existing:

`outputs/e5_final_evaluation/`

The existing E5 result in this workspace uses Trial 5. Because evaluation artifacts were already present, it must not be represented as a fresh blind evaluation.

---

## 7. GrabCut Evaluation

The separate GrabCut evaluation script is:

`evaluate_grabcut.py`

Its output, `outputs/grabcut_eval_metrics.csv`, corresponds to a different rectangle-based GrabCut algorithm.

It should not be confused with the Trial 5 workflow or its metrics.

---

## 8. Egg Tracking Implementation

A separate tracking implementation is available:

`src/track_eggs_kalman_hungarian.py`

The implementation uses:

- Kalman filtering
- Hungarian assignment

Generated tracking artifacts are stored locally under:

`outputs/tracking/`

The current workspace includes:

- `corrected_tracked_eggs.mp4`
- `corrected_tracks.csv`

These are generated output artifacts and are excluded from the GitHub source-code submission.

The existence of tracking code and generated outputs does not, by itself, establish perfect tracking performance. Identity continuity and tracking quality should be evaluated separately before making claims about end-to-end tracking accuracy.

---

## 9. Tests and Verification

The unit tests are located under:

`tests/`

Run the test suite:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The previously executed test suite passed all six tests.

Run the workflow audit:

```bash
.venv/bin/python src/egg_workflow.py audit
```

Run the workflow stages:

```bash
.venv/bin/python src/egg_workflow.py e0
.venv/bin/python src/egg_workflow.py e1
.venv/bin/python src/egg_workflow.py e2
.venv/bin/python src/egg_workflow.py e4
.venv/bin/python src/egg_workflow.py e5
```

E3 is part of the workflow output structure; the commands above reflect the stages used in the documented execution sequence.

---

## 10. Repository and Submission Structure

The GitHub repository is intended to contain the source code, tests, documentation, and configuration needed to understand the implementation.

The following local assets are excluded from the repository:

- Dataset images under `dataset/`
- Generated outputs under `outputs/`
- SAM checkpoint files under `models/`
- Generated tracking videos and CSV files
- Virtual environment and Python cache files

The `.gitignore` file contains the corresponding exclusions.

The local workspace retains the dataset, model checkpoint, pseudo-labels, and generated outputs for experimentation and reproducibility.

---

## 11. Limitations and Honest Reporting

The following limitations should be considered when interpreting the results:

1. SAM masks are pseudo-labels, not independently verified ground truth.
2. The custom development/evaluation split crosses the original dataset folder boundaries.
3. Nine images from the original folder named `test` are included in the custom development split.
4. The existing evaluation artifacts mean the current evaluation cannot be described as a pristine blind test.
5. Trial 8's marginal improvement over Trial 5 is negligible.
6. The presence of tracking code and generated tracking artifacts does not establish perfect identity continuity.
7. Generated outputs and large reference assets are excluded from the GitHub repository.

## 12. Conclusion

Trial 5 remains the selected classical segmentation baseline after comparison with Trials 6–10.

The additive E0–E5 workflow reproduces the Trial 5 development results across all 88 development images, with a maximum IoU difference of approximately 0.0000005.

The project includes segmentation experiments, an auditable workflow, unit tests, and a separate Kalman-filter/Hungarian-assignment tracking implementation.

All reported segmentation metrics are based on agreement with SAM pseudo-labels. The evaluation limitations and tracking validation requirements should be retained when presenting the project or discussing its performance.