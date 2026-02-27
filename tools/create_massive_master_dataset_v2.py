"""
MassiveMasterDataset V2 — Corrected Generator.

Fixes 3 critical bugs from V1:
  1. ROOF3D: Decodes uncompressed RLE annotations (not polygons).
  2. RID: Correct class mapping (val 5=bg, val 4=flat, vals 0-3=sloped)
     + overlays superstructures layer (PVModule→Panel, Chimney/Dormer/etc→Obstacle).
  3. SODwS-V1: Extracts solar panel masks (binary from RGB channel).

Augments all images with 4 rotations (0°, 90°, 180°, 270°) and resizes to 512×512.
Skips any image whose mask is 100% background.
"""

import os
import cv2
import json
import base64
import zlib
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- Paths ---
OUT_DIR = Path('/Users/voskan/Desktop/DeepRoof-2026/data/MassiveMasterDataset')
RID_DIR = Path('/Users/voskan/roofscope_data/roof_information_dataset_2')
ROOF3D_DIR = Path('/Users/voskan/roofscope_data/ROOF3D')
NINJA_DIR = Path('/Users/voskan/roofscope_data/semantic-segmentation-satellite-imagery-DatasetNinja')
YOLO_DIR = Path('/Users/voskan/roofscope_data/yolo_satellite')
SODWS_DIR = Path('/Users/voskan/roofscope_data/SODwS-V1')

TARGET_SIZE = (512, 512)

# --- Helpers ---
def setup_dirs():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    (OUT_DIR / 'images').mkdir(parents=True)
    (OUT_DIR / 'masks').mkdir(parents=True)


def apply_rotations(img, mask, base_name):
    pairs = [
        (img, mask, f"{base_name}_rot0"),
        (cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE), f"{base_name}_rot90"),
        (cv2.rotate(img, cv2.ROTATE_180), cv2.rotate(mask, cv2.ROTATE_180), f"{base_name}_rot180"),
        (cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE), f"{base_name}_rot270"),
    ]
    return pairs


def save_pair(img, mask, name):
    """Resize to TARGET_SIZE, skip if mask is 100% background, save."""
    if img is None or mask is None:
        return 0
    img_r = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    mask_r = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    if mask_r.max() == 0:
        return 0
    cv2.imwrite(str(OUT_DIR / 'images' / f'{name}.jpg'), img_r)
    cv2.imwrite(str(OUT_DIR / 'masks' / f'{name}.png'), mask_r, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return 1


def save_augmented(img, mask, base_name):
    count = 0
    for im, mk, nm in apply_rotations(img, mask, base_name):
        count += save_pair(im, mk, nm)
    return count


# --- RLE Decoder for ROOF3D ---
def decode_uncompressed_rle(counts, h, w):
    """Decode uncompressed RLE (COCO format: alternating bg/fg run-lengths)."""
    mask = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for i, c in enumerate(counts):
        if i % 2 == 1:  # odd index = foreground
            mask[pos:pos + c] = 1
        pos += c
    return mask.reshape((h, w), order='F')  # COCO uses Fortran (column-major) order


# --- 1. ROOF3D Extraction ---
def extract_roof3d():
    print("\n=== Extracting ROOF3D (RLE COCO) ===")
    ann_path = ROOF3D_DIR / 'train' / 'annotation_plane.json'
    img_dir = ROOF3D_DIR / 'train' / 'rgb'
    if not ann_path.exists():
        print("  ROOF3D annotation not found, skipping.")
        return 0

    with open(ann_path) as f:
        coco = json.load(f)

    img_dict = {img['id']: img for img in coco['images']}
    # Group annotations by image_id
    ann_by_img = {}
    for ann in coco['annotations']:
        iid = ann['image_id']
        ann_by_img.setdefault(iid, []).append(ann)

    total = 0
    for img_id, info in tqdm(img_dict.items(), desc="ROOF3D"):
        img_path = img_dir / info['file_name']
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = info['height'], info['width']
        mask = np.zeros((h, w), dtype=np.uint8)

        for ann in ann_by_img.get(img_id, []):
            seg = ann.get('segmentation', [])
            # Handle RLE (either direct dict or list of dicts)
            if isinstance(seg, dict) and 'counts' in seg:
                rle_mask = decode_uncompressed_rle(seg['counts'], seg['size'][0], seg['size'][1])
                mask[rle_mask > 0] = 2  # Sloped Roof
            elif isinstance(seg, list):
                for s in seg:
                    if isinstance(s, dict) and 'counts' in s:
                        rle_mask = decode_uncompressed_rle(s['counts'], s['size'][0], s['size'][1])
                        mask[rle_mask > 0] = 2  # Sloped Roof
                    elif isinstance(s, list) and len(s) >= 6:
                        poly = np.array(s).reshape((-1, 2)).astype(np.int32)
                        cv2.fillPoly(mask, [poly], 2)

        total += save_augmented(img, mask, f"roof3d_{Path(info['file_name']).stem}")

    print(f"  ROOF3D: {total} images saved.")
    return total


# --- 2. RID Extraction (Corrected) ---
def extract_rid():
    print("\n=== Extracting RID (Dual-Layer) ===")
    img_dir = RID_DIR / 'images'
    seg_dir = RID_DIR / 'masks' / 'masks_segments'
    sup_dir = RID_DIR / 'masks' / 'masks_superstructures'

    if not img_dir.exists() or not seg_dir.exists():
        print("  RID paths not found, skipping.")
        return 0

    files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.tif', '.jpg'))])
    total = 0

    for f in tqdm(files, desc="RID"):
        img = cv2.imread(str(img_dir / f))
        if img is None:
            continue

        # --- Segments layer ---
        seg_path = seg_dir / f
        if not seg_path.exists():
            seg_path = seg_dir / f.replace('.tif', '.png').replace('.jpg', '.png')
        seg = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED) if seg_path.exists() else None

        h, w = img.shape[:2]
        unified = np.zeros((h, w), dtype=np.uint8)

        if seg is not None:
            # RID segments: value 5 = background, value 4 = flat roof, values 0-3 = sloped facets
            unified[np.isin(seg, [0, 1, 2, 3])] = 2  # Sloped Roof
            unified[seg == 4] = 1  # Flat Roof
            # value 5 stays as 0 (background)

        # --- Superstructures layer (overlay) ---
        sup_path = sup_dir / f
        if not sup_path.exists():
            sup_path = sup_dir / f.replace('.tif', '.png').replace('.jpg', '.png')
        sup = cv2.imread(str(sup_path), cv2.IMREAD_UNCHANGED) if sup_path.exists() else None

        if sup is not None:
            # Superstructures: value 5 = background (ignore), values 0-4 = superstructure objects
            # We map ALL non-background superstructure pixels to Obstacle (class 4)
            # since we can't distinguish PVModule from Chimney by pixel value alone
            sup_fg = (sup != 5) & (sup != 255)  # non-background
            unified[sup_fg] = 4  # Roof Obstacle

        total += save_augmented(img, unified, f"rid_{Path(f).stem}")

    print(f"  RID: {total} images saved.")
    return total


# --- 3. DatasetNinja Extraction ---
def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.frombuffer(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if len(mask.shape) == 3 and mask.shape[2] == 4:
        return mask[:, :, 3] > 0
    elif len(mask.shape) == 3:
        return mask[:, :, 0] > 0
    return mask > 0


def extract_ninja():
    print("\n=== Extracting DatasetNinja ===")
    ninja_map = {
        'solar panels': 3, 'chimney': 4, 'satellite antenna': 4,
        'window': 4, 'secondary structure': 4, 'property roof': 2,
    }
    total = 0
    for split in ['train', 'val']:
        ann_dir = NINJA_DIR / split / 'ann'
        img_dir = NINJA_DIR / split / 'img'
        if not ann_dir.exists():
            continue

        for ann_name in tqdm(sorted(os.listdir(ann_dir)), desc=f"Ninja {split}"):
            if not ann_name.endswith('.json'):
                continue
            img_name = ann_name.replace('.json', '')
            img_path = img_dir / img_name
            if not img_path.exists():
                continue

            try:
                with open(ann_dir / ann_name) as jf:
                    data = json.load(jf)
            except Exception:
                continue

            h, w = data['size']['height'], data['size']['width']
            mask = np.zeros((h, w), dtype=np.uint8)
            has_labels = False

            # Sort: roof first, then obstacles, then panels (later overwrites earlier)
            objs = sorted(data.get('objects', []),
                          key=lambda o: (0 if o.get('classTitle') == 'property roof' else
                                         1 if o.get('classTitle') in ('chimney', 'satellite antenna', 'window', 'secondary structure') else
                                         2))
            for obj in objs:
                title = obj.get('classTitle', '')
                if title not in ninja_map:
                    continue
                bmp = obj.get('bitmap', {})
                if 'data' not in bmp:
                    continue
                bitmask = base64_2_mask(bmp['data'])
                if bitmask is None:
                    continue
                ox, oy = bmp['origin']
                bh, bw = bitmask.shape
                ye, xe = min(oy + bh, h), min(ox + bw, w)
                cropped = bitmask[0:(ye - oy), 0:(xe - ox)]
                mask[oy:ye, ox:xe][cropped > 0] = ninja_map[title]
                has_labels = True

            if has_labels:
                img = cv2.imread(str(img_path))
                total += save_augmented(img, mask, f"ninja_{Path(img_name).stem}")

    print(f"  Ninja: {total} images saved.")
    return total


# --- 4. YOLO Satellite Extraction ---
def extract_yolo():
    print("\n=== Extracting YOLO Satellite ===")
    yolo_map = {14: 3, 2: 4, 12: 4, 24: 4, 10: 2}
    total = 0

    for split in ['train', 'val']:
        img_dir = YOLO_DIR / 'images' / split
        lbl_dir = YOLO_DIR / 'labels' / split
        if not img_dir.exists():
            continue

        for img_name in tqdm(sorted(os.listdir(img_dir)), desc=f"YOLO {split}"):
            if not img_name.endswith(('.jpg', '.png')):
                continue
            lbl_name = img_name.rsplit('.', 1)[0] + '.txt'
            lbl_path = lbl_dir / lbl_name
            if not lbl_path.exists():
                continue

            img = cv2.imread(str(img_dir / img_name))
            if img is None:
                continue
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = int(parts[0])
                    if cls_id not in yolo_map:
                        continue
                    if len(parts) > 5:  # polygon
                        pts = np.array([float(p) for p in parts[1:]]).reshape(-1, 2)
                        pts[:, 0] *= w
                        pts[:, 1] *= h
                        cv2.fillPoly(mask, [pts.astype(np.int32)], yolo_map[cls_id])
                    else:  # bbox
                        cx, cy, bw2, bh2 = float(parts[1]) * w, float(parts[2]) * h, float(parts[3]) * w, float(parts[4]) * h
                        x1, y1 = int(cx - bw2 / 2), int(cy - bh2 / 2)
                        x2, y2 = int(cx + bw2 / 2), int(cy + bh2 / 2)
                        cv2.rectangle(mask, (x1, y1), (x2, y2), yolo_map[cls_id], -1)

            total += save_augmented(img, mask, f"yolo_{Path(img_name).stem}")

    print(f"  YOLO: {total} images saved.")
    return total


# --- 5. SODwS-V1 Extraction ---
def extract_sodws():
    print("\n=== Extracting SODwS-V1 (Solar Panels) ===")
    total = 0

    for loc in ['Location_A', 'Location_B']:
        img_base = SODWS_DIR / loc / 'images'
        mask_base = SODWS_DIR / loc / 'masks'
        if not img_base.exists() or not mask_base.exists():
            continue

        for orient in sorted(os.listdir(mask_base)):
            mask_orient_dir = mask_base / orient
            img_orient_dir = img_base / orient
            if not mask_orient_dir.is_dir() or not img_orient_dir.is_dir():
                continue

            mask_files = sorted([f for f in os.listdir(mask_orient_dir) if f.endswith(('.png', '.jpg', '.tif'))])
            for mf in tqdm(mask_files, desc=f"SODwS {loc}/{orient}"):
                mask_raw = cv2.imread(str(mask_orient_dir / mf), cv2.IMREAD_UNCHANGED)
                if mask_raw is None:
                    continue

                # Find matching image
                img_path = img_orient_dir / mf
                if not img_path.exists():
                    # Try different extensions
                    for ext in ['.png', '.jpg', '.tif']:
                        candidate = img_orient_dir / (Path(mf).stem + ext)
                        if candidate.exists():
                            img_path = candidate
                            break
                    else:
                        continue

                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Convert raw mask to binary solar panel mask
                if len(mask_raw.shape) == 3:
                    # Use any channel — panel pixels are bright
                    gray = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2GRAY)
                else:
                    gray = mask_raw

                unified = np.zeros(gray.shape, dtype=np.uint8)
                unified[gray > 128] = 3  # Solar Panel

                total += save_augmented(img, unified, f"sodws_{loc}_{orient}_{Path(mf).stem}")

    print(f"  SODwS: {total} images saved.")
    return total


# --- Main ---
if __name__ == '__main__':
    setup_dirs()

    t1 = extract_roof3d()
    t2 = extract_rid()
    t3 = extract_ninja()
    t4 = extract_yolo()
    t5 = extract_sodws()

    total = t1 + t2 + t3 + t4 + t5

    # Generate train.txt
    print("\nGenerating train.txt...")
    mask_stems = sorted([f[:-4] for f in os.listdir(OUT_DIR / 'masks') if f.endswith('.png')])
    with open(OUT_DIR / 'train.txt', 'w') as f:
        for s in mask_stems:
            f.write(s + '\n')

    print(f"\n{'=' * 50}")
    print(f"MASSIVE DATASET V2 GENERATION COMPLETE!")
    print(f"Total images saved: {total}")
    print(f"train.txt entries: {len(mask_stems)}")
    print(f"{'=' * 50}")
