# DeepRoof-2026 Recovery Plan And Execution

Date: 2026-03-03

## 1) What Had To Be Done (Task List)
- [x] Audit all source datasets in `/Users/voskan/roofscope_data` (integrity, labels, class semantics, pair matching).
- [x] Audit current merged dataset `data/MassiveMasterDataset` (split leakage, class balance, file integrity).
- [x] Fix dataset builder logic (RID class mapping, YOLO class mapping, SODwS mask binarization).
- [x] Rebuild merged dataset with corrected mapping.
- [x] Rebuild train/val split correctly (group by base sample, no rotation leakage, no hash leakage).
- [x] Fix training-target generation in dataset class (`DeepRoofDataset`) to remove invalid background instance supervision.
- [x] Fix facet metric to be foreground-only and class-aware.
- [x] Fix train CLI compatibility with actual run commands (`--config`, `--gpus` compatibility).
- [x] Sync source-of-truth/docs class count and canonical commands.
- [x] Update training notebook checkpoint/validation cadence and best metric key.
- [ ] Start new long training run on cleaned dataset and attach new learning curves.

## 2) What Was Found (Root Causes)
### 2.1 Source datasets audit
- RID (`roof_information_dataset_2`):
  - 4764 images, segment and superstructure masks fully paired.
  - Segment mask values: `{0,1,2,3,4,5}`.
  - Superstructure mask values: `{0,1,2,3,4,5}`.
- ROOF3D:
  - 3337 images, 364775 annotations, all segmentations are list-of-RLE-dicts.
  - No missing files, no size mismatch.
- DatasetNinja:
  - 261 annotation files, 2478 objects, all paired with images.
  - Important classes: `property roof=222`, `solar panels=10`, `chimney=150`, `window=120`, `secondary structure=144`.
- YOLO satellite:
  - 261 label files, all paired.
  - Important classes exist in `data.yaml` (`property roof=10`, `solar panels=14`, `chimney=2`, `satellite antenna=12`, `secondary structure=13`, `window=24`).

### 2.2 Merged dataset audit (before fix)
- Total: 33600 samples (`train=30240`, `val=3360`).
- Severe leakage:
  - `exact_overlap=0` but `base_overlap=2912/2912` (all val base scenes leaked into train via rotations).
  - `hash_leakage=5` identical masks across train/val.
- Class imbalance before rebuild:
  - `{bg=77.318%, flat=3.863%, sloped=17.054%, panel=0.469%, obstacle=1.295%}`.
- Critical training-target bug:
  - `DeepRoofDataset` created a giant background instance and included it in `gt_instances`.
- Critical metric bug:
  - `DeepRoofFacetMetric` did not filter background and ignored class labels in matching.

## 3) What Was Changed
### 3.1 Dataset builder and split logic
File: `tools/create_massive_master_dataset_v2.py`
- Added robust CLI mode (`--split-only`, `--val-ratio`, `--seed`, leakage control).
- Fixed RID superstructure mapping using audited semantics:
  - `sup value 0 -> class 3 (solar_panel)`.
  - `sup values 1..4 -> class 4 (roof_obstacle)`.
  - `sup value 5 -> background`.
- Improved YOLO mapping by parsing `data.yaml` labels (not brittle hardcoded subset).
- Improved SODwS binarization with Otsu thresholding.
- Added rotation-safe split by base-id and hash-leakage elimination.
- Added automatic split and class summary reporting.

### 3.2 Dataset target generation
File: `deeproof/datasets/roof_dataset.py`
- Removed `panel -> background` remap.
- Removed synthetic background instance from instance supervision.
- Enabled instance CC generation for class `3` (solar panels).
- Kept class-specific slope-based splitting only for class `2` (sloped roofs).

### 3.3 Evaluation metric fixes
File: `deeproof/evaluation/metrics.py`
- Added label extraction helper.
- Filtered out background (`label=0`) from pred/gt for facet evaluation.
- Added class-aware IoU matching (mismatched classes set IoU=0).

### 3.4 Training/runtime config and tooling
- File: `configs/deeproof_production_swin_L.py`
  - Adjusted class weights to train panel/obstacle classes (`[1.0, 5.0, 1.0, 9.0, 10.0, 0.1]`).
  - Reduced `val_interval` and checkpoint interval from `5000` to `2000`.
- File: `tools/train.py`
  - Added compatibility for both positional config and `--config`.
  - Added `--gpus` compatibility flag with warning (no silent crash).
- File: `notebooks/train_deeproof.ipynb`
  - Updated checkpoint interval to `2000`.
  - Updated best metric to `facet/AP50`.
  - Updated notebook to set `cfg.train_cfg.val_interval = 2000`.
- Files: `docs/source_of_truth.json`, `README.md`, `prd.md`
  - Synced `num_classes=5` and canonical train command.

## 4) Rebuild Results (After Fix)
Rebuilt with:
`python tools/create_massive_master_dataset_v2.py --out-dir data/MassiveMasterDataset --val-ratio 0.1 --seed 42`

### Final merged dataset
- Total samples: `33752`
- Split: `train=30392`, `val=3360`
- Leakage checks:
  - `exact_overlap=0`
  - `base_overlap=0`
  - `hash_leakage=0`

### Source composition (after rebuild)
- Train: `{'ninja': 828, 'rid': 17160, 'roof3d': 11100, 'sodws': 364, 'yolo': 940}`
- Val: `{'ninja': 92, 'rid': 1896, 'roof3d': 1232, 'sodws': 36, 'yolo': 104}`

### Class distribution (after rebuild)
- `{bg=76.518%, flat=3.846%, sloped=16.977%, panel=0.921%, obstacle=1.738%}`
- Panel class increased from `0.469%` to `0.921%` (about 2x).

## 5) Verification
Executed:
`pytest -q tests/test_roof_dataset.py tests/test_train_step_integration.py tests/test_inference_regression.py tests/test_sync_docs.py`

Result:
- `7 passed, 1 skipped`

## 6) Remaining Step
- [ ] Launch fresh training from step 0 on rebuilt dataset and validate new curves/checkpoint behavior (`val_interval=2000`) with updated metrics.
