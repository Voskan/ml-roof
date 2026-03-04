# DeepRoof Training Degradation: Root-Cause Analysis (2026-03-04)

## Symptoms reported
- Checkpoint quality appears to degrade over time (`best` at later iterations looks worse than early checkpoints).
- Validation metrics are unstable/low despite strong train-loss decrease.
- Example at `iter=2000`: `mIoU=9.43`, `background IoU=0.0`, `facet/AP50=8.00`, `over_seg_rate=41.25`.

## Evidence collected

### 1) Dataset audit (MassiveMasterDataset)
- Train images: `30392`, Val images: `3360`.
- Pixel distribution (full pass):
  - Train: bg `76.50%`, flat `3.88%`, sloped `16.94%`, panel `0.94%`, obstacle `1.74%`.
  - Val: bg `76.65%`, flat `3.55%`, sloped `17.35%`, panel `0.76%`, obstacle `1.69%`.
- Split drift is small; class distributions are consistent.
- Obstacle connected components are highly fragmented (train sampled max `582`, val sampled max `393`), creating many tiny potential false positives.

### 2) Validation pipeline bug
- In `configs/deeproof_production_swin_L.py`, validation dataset had no explicit `test_mode=True`.
- `DeepRoofDataset.__getitem__` applies stochastic augmentation when `test_mode=False`.
- Result: validation used train-time random transforms, making checkpoints non-comparable and noisy.

### 3) Facet metric bias toward degradation
- `DeepRoofFacetMetric` used **all** predicted instances (including low-score tiny masks) for AP/over-seg computation.
- As model learns, extra low-confidence predictions accumulate and can reduce precision/AP, inflating `over_seg_rate`, even when core masks improve.

### 4) Inference/eval instance noise
- `DeepRoofMask2Former.predict()` did not globally filter `pred_instances` by score/area.
- Fallback path had filtering, normal path effectively did not.

### 5) Background supervision gap (already fixed)
- `gt_instances` previously dropped background instances entirely.
- This strongly correlates with `background IoU=0.0` symptom.

## Fixes implemented

### A) Background pseudo-instance in dataset targets
- File: `deeproof/datasets/roof_dataset.py`
- Change: append one background pseudo-instance (`sem_map==0`) into `gt_instances` when present.
- Effect: class 0 receives direct query-level supervision.

### B) Deterministic validation
- File: `configs/deeproof_production_swin_L.py`
- Change: `val_dataloader.dataset.test_mode=True`.
- Effect: disables stochastic train augmentation during val; checkpoint metrics become comparable.

### C) Facet metric prediction filtering
- File: `deeproof/evaluation/metrics.py`
- Added params to `DeepRoofFacetMetric`:
  - `score_thr=0.20`
  - `min_area=64`
  - `max_dets=120`
- Applied filtering before IoU/AP/over-seg/under-seg computation.

### D) Inference instance filtering before evaluation
- File: `deeproof/models/deeproof_model.py`
- Added `_filter_instance_predictions(...)` and applied it in `predict()`.
- Uses `test_cfg` thresholds:
  - `instance_score_thr`
  - `instance_min_area`
  - `max_instances`

### E) Evaluation/inference thresholds configured centrally
- File: `configs/deeproof_production_swin_L.py`
- `model.test_cfg` now includes:
  - `instance_score_thr=0.20`
  - `instance_min_area=64`
  - `max_instances=120`
- `val_evaluator.DeepRoofFacetMetric` aligned to same thresholds.

## Validation of code changes
- `python -m py_compile` passed for modified modules.
- Tests passed:
  - `tests/test_roof_dataset.py`
  - `tests/test_train_step_integration.py`
  - `tests/test_inference_regression.py`
  - `tests/test_sync_docs.py`
  - Result: `7 passed, 1 skipped`.

## What to do next (required)
1. Restart training process (code changes affect dataloader + predict + evaluator).
2. Resume from latest checkpoint (e.g., `iter_30000.pth`).
3. Compare metrics at the next validation points (e.g., 32000/34000) against old run:
   - `mIoU`
   - `background IoU`
   - `facet/AP50`, `facet/AP75`
   - `facet/over_seg_rate`
4. If `background IoU` remains low after these fixes, next target is semantic query fusion calibration (class-0 logit bias in decode inference).
