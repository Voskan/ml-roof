#!/usr/bin/env python3
"""
convert_roofscope.py  —  DeepRoof-2026 Dataset Expansion
=========================================================
Converts external datasets in /Users/voskan/roofscope_data into the
MassiveMasterDataset 10-class format and appends new IDs to train.txt/val.txt.

10-Class Schema
---------------
 0  Background / sky
 1  Flat roof
 2  Sloped roof – South-facing (SSW/S/SSE/SW/SE)   ← best for solar
 3  Solar panel (existing installation)
 4  Obstacle – generic
 5  Chimney
 6  Dormer / Skylight / Window (unusable area)
 7  Sloped roof – North-facing (NNE/N/NNW/NE/NW)   ← worst for solar
 8  Sloped roof – East/West-facing (E/W/ENE/ESE…)   ← medium
 9  AC unit / Mechanical / Other superstructure

Sources
-------
 A) roof_information_dataset_2  -- 4,764 × 512 px  (segments + superstructures)
 B) yolo_satellite               -- 261 images (YOLO bbox → rasterise to mask)
 C) ROOF3D                       -- 3,337 images COCO poly → building footprint
"""

import argparse
import json
import os
import sys
import random
import shutil
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MASTER_ROOT  = PROJECT_ROOT / "data" / "MassiveMasterDataset"
ROOFSCOPE    = Path("/Users/voskan/roofscope_data")

# Destination dirs
DEST_IMAGES  = MASTER_ROOT / "images"
DEST_MASKS   = MASTER_ROOT / "masks"
DEST_IMAGES.mkdir(parents=True, exist_ok=True)
DEST_MASKS.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 512     # All output tiles are 512 × 512

# ─── Class IDs ────────────────────────────────────────────────────────────────
C_BG        = 0
C_FLAT      = 1
C_SLOPED_S  = 2   # south-facing (same as existing class 2)
C_PANEL     = 3
C_OBSTACLE  = 4
C_CHIMNEY   = 5
C_DORMER    = 6
C_SLOPED_N  = 7
C_SLOPED_EW = 8
C_MECH      = 9
IGNORE      = 255

# Cardinal direction → class
SOUTH_LABELS = {"SSW", "S", "SSE", "SW", "SE"}
NORTH_LABELS = {"NNE", "N", "NNW", "NE", "NW"}
# Everything else keeps mapping below; flat → C_FLAT

# ─── Helpers ──────────────────────────────────────────────────────────────────

def safe_resize(img: np.ndarray, size: int = IMG_SIZE, interp_mask: bool = False) -> np.ndarray:
    if img is None:
        return None
    h, w = img.shape[:2]
    if h == size and w == size:
        return img
    interp = cv2.INTER_NEAREST if interp_mask else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)


def save_sample(prefix: str, rgb: np.ndarray, mask: np.ndarray,
                dry_run: bool = False) -> Optional[str]:
    """Save image and mask under DEST dirs. Returns img_id or None on failure."""
    if rgb is None or mask is None:
        return None
    if rgb.shape[:2] != (IMG_SIZE, IMG_SIZE):
        rgb = safe_resize(rgb, IMG_SIZE, interp_mask=False)
    if mask.shape[:2] != (IMG_SIZE, IMG_SIZE):
        mask = safe_resize(mask, IMG_SIZE, interp_mask=True)
    img_path  = DEST_IMAGES / f"{prefix}.jpg"
    mask_path = DEST_MASKS  / f"{prefix}.png"
    if dry_run:
        return prefix
    cv2.imwrite(str(img_path),  rgb)
    cv2.imwrite(str(mask_path), mask.astype(np.uint8))
    return prefix


def load_existing_ids() -> Set[str]:
    ids: Set[str] = set()
    for txt in [MASTER_ROOT / "train.txt", MASTER_ROOT / "val.txt"]:
        if txt.exists():
            with open(txt) as f:
                ids.update(l.strip() for l in f if l.strip())
    return ids


# ─── Source A: roof_information_dataset_2 ─────────────────────────────────────

def _build_seg_label_map(geoms_dir: Path):
    """
    Build {image_name → {segment_index → label_str}} from gdf_all_segments.json.
    Each connected component in masks_segments gets one azimuth label.
    """
    seg_file = geoms_dir / "gdf_all_segments.json"
    if not seg_file.exists():
        return {}
    with open(seg_file) as f:
        data = json.load(f)
    result: Dict[str, Dict[int, str]] = defaultdict(dict)
    for feat in data["features"]:
        props = feat["properties"]
        img_name = props.get("image_name", "")
        idx      = int(props.get("index", -1))
        label    = props.get("label", "flat")
        result[img_name][idx] = label
    return result


def _build_sup_label_map(geoms_dir: Path):
    """
    Build {image_name → {batch_id/index → sup_class_id}} from gdf_all_superstructures.json.
    Superstructure label → our class ID.
    """
    sup_file = geoms_dir / "gdf_all_superstructures.json"
    if not sup_file.exists():
        return {}
    label_map = {
        "Chimney":    C_CHIMNEY,
        "Window":     C_DORMER,
        "Skylight":   C_DORMER,
        "Dormer":     C_DORMER,
        "PVModule":   C_PANEL,
        "AC Outlet":  C_MECH,
        "AC System":  C_MECH,
        "TV-Dish":    C_MECH,
        "Balcony":    C_DORMER,
        "Ladder":     C_MECH,
        "Wall":       C_OBSTACLE,
        "Other":      C_OBSTACLE,
    }
    with open(sup_file) as f:
        data = json.load(f)
    # We don't have pixel-level mapping from JSON to mask value here;
    # the mask already encodes: 5=outside, 0=inside-background (unlabelled),
    # 1=Window, 2=Skylight, 3=Dormer, 4=Chimney/AC/Other
    # Use the fixed mask encoding directly in the conversion below.
    return label_map  # returned for reference; not directly used


# Fixed mask_superstructures encoding (derived empirically)
SUP_MASK_TO_CLASS = {
    0: C_BG,        # inside roof but no superstructure
    1: C_DORMER,    # Window
    2: C_DORMER,    # Skylight
    3: C_DORMER,    # Dormer
    4: C_CHIMNEY,   # Chimney / AC / Other  (most common is chimney by count)
    5: IGNORE,      # outside building footprint
}

def azimuth_label_to_class(label: str) -> int:
    label = label.strip().upper()
    if label == "FLAT":
        return C_FLAT
    if label in SOUTH_LABELS:
        return C_SLOPED_S
    if label in NORTH_LABELS:
        return C_SLOPED_N
    return C_SLOPED_EW


def convert_roof_info(dry_run: bool = False, val_split: float = 0.1,
                      verbose: bool = True) -> Tuple[List[str], List[str]]:
    """Convert roof_information_dataset_2 → MassiveMasterDataset format."""
    src = ROOFSCOPE / "roof_information_dataset_2"
    seg_dir = src / "masks" / "masks_segments"
    sup_dir = src / "masks" / "masks_superstructures"
    img_dir = src / "images"
    geoms   = src / "geometries"

    # Optional: training CSV gives the canonical split
    train_csv = src / "training_split_512.csv"
    test_csv  = src / "test_split_512.csv"
    train_ids_csv: Set[str] = set()
    test_ids_csv:  Set[str] = set()
    if train_csv.exists():
        import csv
        with open(train_csv) as f:
            for row in csv.DictReader(f):
                name = row.get("image_names", "").strip()
                if name:
                    train_ids_csv.add(name)
    if test_csv.exists():
        import csv
        with open(test_csv) as f:
            for row in csv.DictReader(f):
                name = row.get("image_names", "").strip()
                if name:
                    test_ids_csv.add(name)

    all_seg_pngs = sorted(seg_dir.glob("*.png"))
    if verbose:
        print(f"[roof_info] Found {len(all_seg_pngs)} segment masks")

    new_train, new_val = [], []
    errors = 0

    for seg_path in all_seg_pngs:
        stem = seg_path.stem   # e.g. "121453.6_463136.8"

        seg_mask = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
        sup_mask = cv2.imread(str(sup_dir / seg_path.name), cv2.IMREAD_UNCHANGED)
        rgb_path = img_dir / (stem + ".png")
        rgb = cv2.imread(str(rgb_path)) if rgb_path.exists() else None
        if rgb is None:
            errors += 1
            continue
        if seg_mask is None:
            errors += 1
            continue

        # Build unified 10-class mask
        out_mask = np.full((IMG_SIZE, IMG_SIZE), C_BG, dtype=np.uint8)

        # 1) Start from segment mask (azimuth-based facets)
        # Values 1-4 in segment mask are different facets (the JSON index maps to them)
        # We know: value 5 = outside building → IGNORE, value 0 = background
        # For now we map all foreground segment values to C_SLOPED_EW as safe default
        # (more precise per-pixel azimuth mapping would require rasterising polygons)
        seg_fg = (seg_mask > 0) & (seg_mask < 5)
        out_mask[seg_fg] = C_SLOPED_EW  # conservative default

        # Flat roofs: value 0 inside the building bounding box → tricky; skip per-pixel
        # If entire image has only value=5 and value=0 outside → all flat
        uniq = set(seg_mask.flatten().tolist())
        if uniq == {0, 5} or uniq == {5}:
            # Entire image is flat-roof territory or unresolvable
            roof_px = (seg_mask == 0)
            out_mask[roof_px] = C_FLAT

        # value=5 → outside/ignore
        out_mask[seg_mask == 5] = IGNORE

        # 2) Overlay superstructure mask on top (higher priority)
        if sup_mask is not None:
            for raw_val, cls in SUP_MASK_TO_CLASS.items():
                px = (sup_mask == raw_val)
                if raw_val == 5:
                    # outside — use ignore only where segment also says ignore
                    out_mask[px & (seg_mask == 5)] = IGNORE
                elif raw_val > 0:
                    out_mask[px] = cls

        # Resize if needed
        out_mask = safe_resize(out_mask, IMG_SIZE, interp_mask=True)
        rgb       = safe_resize(rgb, IMG_SIZE, interp_mask=False)

        prefix = f"rinfo_{stem.replace('.', 'p').replace(',', '_')}"

        # Determine split
        if stem in test_ids_csv:
            bucket = "val"
        elif stem in train_ids_csv or not test_ids_csv:
            bucket = "train" if (random.random() > val_split) else "val"
        else:
            bucket = "train" if (random.random() > val_split) else "val"

        saved = save_sample(prefix, rgb, out_mask, dry_run=dry_run)
        if saved:
            if bucket == "val":
                new_val.append(saved)
            else:
                new_train.append(saved)

    if verbose:
        print(f"[roof_info] Converted: train={len(new_train)}, val={len(new_val)}, errors={errors}")
    return new_train, new_val


# ─── Source B: yolo_satellite ──────────────────────────────────────────────────

# YOLO label id → our 10-class id  (only relevant classes)
YOLO_CLASS_MAP = {
    2:  C_CHIMNEY,   # chimney
    10: C_OBSTACLE,  # property roof (just mark as generic obstacle overlay)
    12: C_MECH,      # satellite antenna
    14: C_PANEL,     # solar panels
    23: C_MECH,      # water tank / oil tank
    17: C_OBSTACLE,  # swimming pool (obstacle for solar)
    24: C_DORMER,    # window
    16: C_MECH,      # street light
    9:  C_MECH,      # power lines & cables
}

def convert_yolo_satellite(dry_run: bool = False, val_split: float = 0.1,
                            verbose: bool = True) -> Tuple[List[str], List[str]]:
    src = ROOFSCOPE / "yolo_satellite"
    img_root   = src / "images"
    label_root = src / "labels"

    new_train, new_val = [], []
    errors = 0
    all_imgs = list(img_root.rglob("*.jpg")) + list(img_root.rglob("*.png"))

    if verbose:
        print(f"[yolo_sat] Found {len(all_imgs)} images")

    for img_path in all_imgs:
        rgb = cv2.imread(str(img_path))
        if rgb is None:
            errors += 1
            continue
        oh, ow = rgb.shape[:2]

        # Find corresponding label file
        rel = img_path.relative_to(img_root)
        label_path = label_root / rel.parent / (img_path.stem + ".txt")
        if not label_path.exists():
            # Try flat lookup
            label_path = label_root / (img_path.stem + ".txt")
        if not label_path.exists():
            errors += 1
            continue

        # Start with all-background mask
        mask = np.zeros((oh, ow), dtype=np.uint8)

        with open(label_path) as lf:
            for line in lf:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                our_cls = YOLO_CLASS_MAP.get(cls_id)
                if our_cls is None:
                    continue

                # YOLO bbox: cx cy w h (normalised)
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = int((cx - bw / 2) * ow)
                y1 = int((cy - bh / 2) * oh)
                x2 = int((cx + bw / 2) * ow)
                y2 = int((cy + bh / 2) * oh)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(ow, x2), min(oh, y2)
                mask[y1:y2, x1:x2] = our_cls

        rgb  = safe_resize(rgb,  IMG_SIZE, interp_mask=False)
        mask = safe_resize(mask, IMG_SIZE, interp_mask=True)

        prefix = f"yolo_{img_path.stem}"
        bucket = "val" if random.random() < val_split else "train"
        saved = save_sample(prefix, rgb, mask, dry_run=dry_run)
        if saved:
            (new_val if bucket == "val" else new_train).append(saved)

    if verbose:
        print(f"[yolo_sat] Converted: train={len(new_train)}, val={len(new_val)}, errors={errors}")
    return new_train, new_val


# ─── Source C: ROOF3D (building footprint polygons) ───────────────────────────

def _poly_to_mask(poly_pts: list, h: int, w: int) -> np.ndarray:
    """Rasterise a flat polygon list [x0,y0,x1,y1,...] to binary mask."""
    pts = np.array(poly_pts, dtype=np.float32).reshape(-1, 2)
    pts_int = pts.astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_int], 1)
    return mask


def convert_roof3d(dry_run: bool = False, val_split: float = 0.1,
                   verbose: bool = True) -> Tuple[List[str], List[str]]:
    src = ROOFSCOPE / "ROOF3D"

    new_train, new_val = [], []
    errors = 0

    for split_name in ["train", "val"]:
        split_dir = src / split_name
        if not split_dir.exists():
            continue
        jsons = list(split_dir.glob("*.json"))
        tifs  = list(split_dir.glob("*.tif"))

        for jf in jsons:
            with open(jf) as f:
                coco = json.load(f)

            # Build id→file map
            id_to_img = {img["id"]: img for img in coco.get("images", [])}
            # group annotations by image
            ann_by_img = defaultdict(list)
            for ann in coco.get("annotations", []):
                ann_by_img[ann["image_id"]].append(ann)

            for img_id, img_info in id_to_img.items():
                file_name = img_info["file_name"]
                ih = img_info.get("height", IMG_SIZE)
                iw = img_info.get("width", IMG_SIZE)

                # Try to find the actual image
                rgb = None
                for tif in tifs:
                    if tif.stem in file_name or file_name in tif.stem:
                        try:
                            rgb = cv2.imread(str(tif))
                        except Exception:
                            pass
                        break
                # Also try common sub-dirs
                if rgb is None:
                    for candidate in [split_dir / file_name, src / file_name]:
                        if candidate.exists():
                            rgb = cv2.imread(str(candidate))
                            if rgb is None:
                                # try tiff
                                try:
                                    import tifffile
                                    arr = tifffile.imread(str(candidate))
                                    if arr.ndim == 2:
                                        rgb = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                                    elif arr.shape[2] > 3:
                                        rgb = arr[:, :, :3].astype(np.uint8)
                                    else:
                                        rgb = arr.astype(np.uint8)
                                except Exception:
                                    pass
                            if rgb is not None:
                                break

                # Build mask from polygons
                mask = np.zeros((ih, iw), dtype=np.uint8)
                for ann in ann_by_img[img_id]:
                    segs = ann.get("segmentation", [])
                    for seg in segs:
                        if isinstance(seg, list) and len(seg) >= 6:
                            poly_mask = _poly_to_mask(seg, ih, iw)
                            # Mark as C_SLOPED_EW (generic sloped — no orientation info)
                            mask[poly_mask > 0] = C_SLOPED_EW

                if rgb is None:
                    # Create synthetic grey image from mask context if no RGB available
                    errors += 1
                    continue

                rgb  = safe_resize(rgb,  IMG_SIZE, interp_mask=False)
                mask = safe_resize(mask, IMG_SIZE, interp_mask=True)

                prefix = f"roof3d_{split_name}_{img_id}"
                # Honour original split with some val randomisation
                if split_name == "val":
                    bucket = "val"
                else:
                    bucket = "val" if random.random() < val_split else "train"
                saved = save_sample(prefix, rgb, mask, dry_run=dry_run)
                if saved:
                    (new_val if bucket == "val" else new_train).append(saved)

    if verbose:
        print(f"[ROOF3D] Converted: train={len(new_train)}, val={len(new_val)}, errors={errors}")
    return new_train, new_val


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DeepRoof dataset expansion tool")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate conversion without writing files")
    parser.add_argument("--sources", nargs="+",
                        choices=["roof_info", "yolo", "roof3d", "all"],
                        default=["all"],
                        help="Which sources to convert")
    parser.add_argument("--val-split", type=float, default=0.10,
                        help="Fraction of new samples assigned to val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    do_all = "all" in args.sources or not args.sources
    existing = load_existing_ids()
    print(f"Existing MassiveMasterDataset IDs: {len(existing)}")

    all_new_train: List[str] = []
    all_new_val:   List[str] = []

    if do_all or "roof_info" in args.sources:
        t, v = convert_roof_info(dry_run=args.dry_run, val_split=args.val_split,
                                  verbose=args.verbose)
        all_new_train.extend(t)
        all_new_val.extend(v)

    if do_all or "yolo" in args.sources:
        t, v = convert_yolo_satellite(dry_run=args.dry_run, val_split=args.val_split,
                                       verbose=args.verbose)
        all_new_train.extend(t)
        all_new_val.extend(v)

    if do_all or "roof3d" in args.sources:
        t, v = convert_roof3d(dry_run=args.dry_run, val_split=args.val_split,
                               verbose=args.verbose)
        all_new_train.extend(t)
        all_new_val.extend(v)

    # Filter out duplicates
    all_new_train = [x for x in all_new_train if x not in existing]
    all_new_val   = [x for x in all_new_val   if x not in existing]

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}New train samples: {len(all_new_train)}")
    print(f"{'[DRY RUN] ' if args.dry_run else ''}New val   samples: {len(all_new_val)}")

    if not args.dry_run:
        with open(MASTER_ROOT / "train.txt", "a") as f:
            for img_id in all_new_train:
                f.write(img_id + "\n")
        with open(MASTER_ROOT / "val.txt", "a") as f:
            for img_id in all_new_val:
                f.write(img_id + "\n")
        print(f"\nAppended to train.txt ({len(all_new_train)} entries)")
        print(f"Appended to val.txt   ({len(all_new_val)} entries)")
    else:
        print("\n[DRY RUN] No files written. Pass without --dry-run to execute.")

    print("\nDone.")


if __name__ == "__main__":
    main()
