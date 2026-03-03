"""
MassiveMasterDataset V2 — corrected generator and splitter.

Key fixes:
1) ROOF3D: decode uncompressed COCO RLE (list-of-dicts in annotation_plane.json).
2) RID mapping:
   - segments: value 5=bg, 4=flat, 0..3=sloped.
   - superstructures: value 0=PVModule -> class 3 (solar_panel),
     values 1..4 -> class 4 (roof_obstacle), value 5=bg.
   This mapping is inferred from geospatial alignment between
   gdf_images_with_labels_512.json and gdf_all_superstructures.json.
3) SODwS-V1: robust per-image Otsu thresholding for binary panel extraction.
4) Split generation: base-id grouped train/val split (rotation-safe) with
   optional hash-leakage elimination across train/val.

Unified classes:
0=background, 1=flat_roof, 2=sloped_roof, 3=solar_panel, 4=roof_obstacle
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import re
import shutil
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# --- Paths ---
OUT_DIR = Path('/Users/voskan/Desktop/DeepRoof-2026/data/MassiveMasterDataset')
RID_DIR = Path('/Users/voskan/roofscope_data/roof_information_dataset_2')
ROOF3D_DIR = Path('/Users/voskan/roofscope_data/ROOF3D')
NINJA_DIR = Path('/Users/voskan/roofscope_data/semantic-segmentation-satellite-imagery-DatasetNinja')
YOLO_DIR = Path('/Users/voskan/roofscope_data/yolo_satellite')
SODWS_DIR = Path('/Users/voskan/roofscope_data/SODwS-V1')

TARGET_SIZE = (512, 512)
ROT_RE = re.compile(r'_rot(?:0|90|180|270)$')

# RID superstructure map inferred from geometry alignment audit.
RID_SUPERSTRUCTURE_TO_CLASS = {
    0: 3,  # PVModule
    1: 4,
    2: 4,
    3: 4,
    4: 4,
    5: 0,  # background
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create/split MassiveMasterDataset V2')
    parser.add_argument('--out-dir', type=Path, default=OUT_DIR, help='Output dataset directory')
    parser.add_argument(
        '--split-only',
        action='store_true',
        help='Only regenerate train.txt/val.txt from existing masks without re-extraction',
    )
    parser.add_argument('--val-ratio', type=float, default=0.10, help='Validation ratio at base-id level')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for split generation')
    parser.add_argument(
        '--allow-hash-leakage',
        action='store_true',
        help='Do not eliminate identical-mask leakage across train/val',
    )
    return parser.parse_args()


# --- Helpers ---
def setup_dirs(out_dir: Path):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / 'images').mkdir(parents=True)
    (out_dir / 'masks').mkdir(parents=True)


def apply_rotations(img: np.ndarray, mask: np.ndarray, base_name: str):
    pairs = [
        (img, mask, f'{base_name}_rot0'),
        (
            cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE),
            f'{base_name}_rot90',
        ),
        (cv2.rotate(img, cv2.ROTATE_180), cv2.rotate(mask, cv2.ROTATE_180), f'{base_name}_rot180'),
        (
            cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
            cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE),
            f'{base_name}_rot270',
        ),
    ]
    return pairs


def save_pair(out_dir: Path, img: np.ndarray, mask: np.ndarray, name: str) -> int:
    """Resize to TARGET_SIZE, skip 100% background, save image/mask pair."""
    if img is None or mask is None:
        return 0
    img_r = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    mask_r = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    if mask_r.max() == 0:
        return 0
    cv2.imwrite(str(out_dir / 'images' / f'{name}.jpg'), img_r)
    cv2.imwrite(str(out_dir / 'masks' / f'{name}.png'), mask_r, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return 1


def save_augmented(out_dir: Path, img: np.ndarray, mask: np.ndarray, base_name: str) -> int:
    count = 0
    for im, mk, nm in apply_rotations(img, mask, base_name):
        count += save_pair(out_dir, im, mk, nm)
    return count


def _find_with_fallback(dir_path: Path, file_name: str) -> Path | None:
    p = dir_path / file_name
    if p.exists():
        return p
    stem = Path(file_name).stem
    for ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
        c = dir_path / f'{stem}{ext}'
        if c.exists():
            return c
    return None


def _sample_base_id(sample_id: str) -> str:
    return ROT_RE.sub('', sample_id)


def _sample_source(sample_id: str) -> str:
    return sample_id.split('_', 1)[0]


# --- RLE Decoder for ROOF3D ---
def decode_uncompressed_rle(counts: list[int], h: int, w: int) -> np.ndarray:
    """Decode uncompressed RLE (COCO format: alternating bg/fg runs)."""
    mask = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for i, c in enumerate(counts):
        if i % 2 == 1:
            mask[pos:pos + c] = 1
        pos += c
    return mask.reshape((h, w), order='F')


# --- 1. ROOF3D Extraction ---
def extract_roof3d(out_dir: Path) -> int:
    print('\n=== Extracting ROOF3D (RLE COCO) ===')
    ann_path = ROOF3D_DIR / 'train' / 'annotation_plane.json'
    img_dir = ROOF3D_DIR / 'train' / 'rgb'
    if not ann_path.exists():
        print('  ROOF3D annotation not found, skipping.')
        return 0

    with open(ann_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    img_dict = {img['id']: img for img in coco['images']}
    ann_by_img: dict[int, list[dict]] = defaultdict(list)
    for ann in coco['annotations']:
        ann_by_img[int(ann['image_id'])].append(ann)

    total = 0
    for img_id, info in tqdm(img_dict.items(), desc='ROOF3D'):
        img_path = img_dir / info['file_name']
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = int(info['height']), int(info['width'])
        mask = np.zeros((h, w), dtype=np.uint8)

        for ann in ann_by_img.get(int(img_id), []):
            seg = ann.get('segmentation', [])
            if isinstance(seg, dict) and 'counts' in seg and isinstance(seg.get('counts'), list):
                rle_mask = decode_uncompressed_rle(seg['counts'], int(seg['size'][0]), int(seg['size'][1]))
                mask[rle_mask > 0] = 2
            elif isinstance(seg, list):
                for s in seg:
                    if isinstance(s, dict) and 'counts' in s and isinstance(s.get('counts'), list):
                        rle_mask = decode_uncompressed_rle(s['counts'], int(s['size'][0]), int(s['size'][1]))
                        mask[rle_mask > 0] = 2
                    elif isinstance(s, list) and len(s) >= 6:
                        poly = np.array(s).reshape((-1, 2)).astype(np.int32)
                        cv2.fillPoly(mask, [poly], 2)

        total += save_augmented(out_dir, img, mask, f"roof3d_{Path(info['file_name']).stem}")

    print(f'  ROOF3D: {total} images saved.')
    return total


# --- 2. RID Extraction ---
def extract_rid(out_dir: Path) -> int:
    print('\n=== Extracting RID (Dual-Layer) ===')
    img_dir = RID_DIR / 'images'
    seg_dir = RID_DIR / 'masks' / 'masks_segments'
    sup_dir = RID_DIR / 'masks' / 'masks_superstructures'

    if not img_dir.exists() or not seg_dir.exists() or not sup_dir.exists():
        print('  RID paths not found, skipping.')
        return 0

    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.tif', '.jpg', '.jpeg'))])
    total = 0

    for fname in tqdm(files, desc='RID'):
        img = cv2.imread(str(img_dir / fname))
        if img is None:
            continue

        seg_path = _find_with_fallback(seg_dir, fname)
        sup_path = _find_with_fallback(sup_dir, fname)

        seg = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED) if seg_path else None
        sup = cv2.imread(str(sup_path), cv2.IMREAD_UNCHANGED) if sup_path else None

        h, w = img.shape[:2]
        unified = np.zeros((h, w), dtype=np.uint8)

        if seg is not None:
            unified[np.isin(seg, [0, 1, 2, 3])] = 2  # sloped
            unified[seg == 4] = 1  # flat

        if sup is not None:
            # precise mapping derived from geospatial alignment audit.
            for src_val, dst_cls in RID_SUPERSTRUCTURE_TO_CLASS.items():
                if dst_cls == 0:
                    continue
                unified[sup == src_val] = dst_cls

        total += save_augmented(out_dir, img, unified, f'rid_{Path(fname).stem}')

    print(f'  RID: {total} images saved.')
    return total


# --- 3. DatasetNinja Extraction ---
def base64_2_mask(s: str) -> np.ndarray | None:
    z = zlib.decompress(base64.b64decode(s))
    n = np.frombuffer(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if len(mask.shape) == 3 and mask.shape[2] == 4:
        return mask[:, :, 3] > 0
    if len(mask.shape) == 3:
        return mask[:, :, 0] > 0
    return mask > 0


def extract_ninja(out_dir: Path) -> int:
    print('\n=== Extracting DatasetNinja ===')
    ninja_map = {
        'solar panels': 3,
        'chimney': 4,
        'satellite antenna': 4,
        'window': 4,
        'secondary structure': 4,
        'property roof': 2,
    }
    total = 0

    for split in ['train', 'val']:
        ann_dir = NINJA_DIR / split / 'ann'
        img_dir = NINJA_DIR / split / 'img'
        if not ann_dir.exists() or not img_dir.exists():
            continue

        for ann_name in tqdm(sorted(os.listdir(ann_dir)), desc=f'Ninja {split}'):
            if not ann_name.endswith('.json'):
                continue
            img_name = ann_name.replace('.json', '')
            img_path = img_dir / img_name
            if not img_path.exists():
                continue

            try:
                with open(ann_dir / ann_name, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
            except Exception:
                continue

            h, w = int(data['size']['height']), int(data['size']['width'])
            mask = np.zeros((h, w), dtype=np.uint8)
            has_labels = False

            # roof first, then obstacles, then panels (panels overwrite roof)
            objs = sorted(
                data.get('objects', []),
                key=lambda o: (
                    0 if o.get('classTitle') == 'property roof' else
                    1 if o.get('classTitle') in ('chimney', 'satellite antenna', 'window', 'secondary structure') else
                    2
                ),
            )

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

                ox, oy = bmp.get('origin', [0, 0])
                ox, oy = int(ox), int(oy)
                bh, bw = bitmask.shape
                ye, xe = min(oy + bh, h), min(ox + bw, w)
                if ye <= oy or xe <= ox:
                    continue
                cropped = bitmask[0:(ye - oy), 0:(xe - ox)]
                mask[oy:ye, ox:xe][cropped > 0] = ninja_map[title]
                has_labels = True

            if has_labels:
                img = cv2.imread(str(img_path))
                total += save_augmented(out_dir, img, mask, f'ninja_{Path(img_name).stem}')

    print(f'  Ninja: {total} images saved.')
    return total


# --- 4. YOLO Satellite Extraction ---
def _load_yolo_mapping() -> dict[int, int]:
    """Read class names from data.yaml and map to unified classes."""
    yaml_path = YOLO_DIR / 'data.yaml'
    default_map = {14: 3, 2: 4, 12: 4, 13: 4, 24: 4, 10: 2}
    if not yaml_path.exists():
        return default_map

    names: dict[int, str] = {}
    try:
        lines = yaml_path.read_text(encoding='utf-8').splitlines()
        in_names = False
        for line in lines:
            raw = line.strip()
            if not raw:
                continue
            if raw.startswith('names:'):
                in_names = True
                continue
            if not in_names:
                continue
            if re.match(r'^\d+\s*:\s*.+$', raw):
                idx, name = raw.split(':', 1)
                names[int(idx.strip())] = name.strip().lower()
    except Exception:
        return default_map

    if not names:
        return default_map

    out: dict[int, int] = {}
    for idx, name in names.items():
        if 'solar' in name or 'pv' in name:
            out[idx] = 3
        elif name in {'property roof', 'roof'} or 'roof' in name:
            out[idx] = 2
        elif any(k in name for k in (
            'chimney',
            'satellite antenna',
            'window',
            'secondary structure',
            'skylight',
            'dormer',
            'balcony',
            'ac',
            'dish',
            'ladder',
            'wall',
            'tank',
        )):
            out[idx] = 4
    return out if out else default_map


def extract_yolo(out_dir: Path) -> int:
    print('\n=== Extracting YOLO Satellite ===')
    yolo_map = _load_yolo_mapping()
    print(f'  YOLO mapped class ids: {sorted(yolo_map.keys())}')
    total = 0

    for split in ['train', 'val']:
        img_dir = YOLO_DIR / 'images' / split
        lbl_dir = YOLO_DIR / 'labels' / split
        if not img_dir.exists() or not lbl_dir.exists():
            continue

        for img_name in tqdm(sorted(os.listdir(img_dir)), desc=f'YOLO {split}'):
            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff')):
                continue
            lbl_name = f"{Path(img_name).stem}.txt"
            lbl_path = lbl_dir / lbl_name
            if not lbl_path.exists():
                continue

            img = cv2.imread(str(img_dir / img_name))
            if img is None:
                continue
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

            with open(lbl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cls_id = int(float(parts[0]))
                    except Exception:
                        continue
                    if cls_id not in yolo_map:
                        continue

                    if len(parts) > 5:
                        pts = np.array([float(p) for p in parts[1:]], dtype=np.float32).reshape(-1, 2)
                        pts[:, 0] *= w
                        pts[:, 1] *= h
                        cv2.fillPoly(mask, [pts.astype(np.int32)], int(yolo_map[cls_id]))
                    elif len(parts) == 5:
                        cx, cy, bw2, bh2 = (
                            float(parts[1]) * w,
                            float(parts[2]) * h,
                            float(parts[3]) * w,
                            float(parts[4]) * h,
                        )
                        x1, y1 = int(cx - bw2 / 2), int(cy - bh2 / 2)
                        x2, y2 = int(cx + bw2 / 2), int(cy + bh2 / 2)
                        cv2.rectangle(mask, (x1, y1), (x2, y2), int(yolo_map[cls_id]), -1)

            total += save_augmented(out_dir, img, mask, f'yolo_{Path(img_name).stem}')

    print(f'  YOLO: {total} images saved.')
    return total


# --- 5. SODwS-V1 Extraction ---
def extract_sodws(out_dir: Path) -> int:
    print('\n=== Extracting SODwS-V1 (Solar Panels) ===')
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

            mask_files = sorted(
                [f for f in os.listdir(mask_orient_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
            )
            for mf in tqdm(mask_files, desc=f'SODwS {loc}/{orient}'):
                mask_raw = cv2.imread(str(mask_orient_dir / mf), cv2.IMREAD_UNCHANGED)
                if mask_raw is None:
                    continue

                img_path = _find_with_fallback(img_orient_dir, mf)
                if img_path is None:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                gray = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2GRAY) if mask_raw.ndim == 3 else mask_raw
                # Robustly binarize masks that were saved with compression artifacts.
                otsu_thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thr = max(16.0, float(otsu_thr))
                unified = np.zeros(gray.shape, dtype=np.uint8)
                unified[gray > thr] = 3

                total += save_augmented(out_dir, img, unified, f'sodws_{loc}_{orient}_{Path(mf).stem}')

    print(f'  SODwS: {total} images saved.')
    return total


# --- Split helpers ---
def _mask_file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def _build_split(
    out_dir: Path,
    mask_stems: list[str],
    val_ratio: float = 0.10,
    seed: int = 42,
    eliminate_hash_leakage: bool = True,
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)

    base_to_samples: dict[str, list[str]] = defaultdict(list)
    for sid in mask_stems:
        base_to_samples[_sample_base_id(sid)].append(sid)

    source_to_bases: dict[str, list[str]] = defaultdict(list)
    for base in base_to_samples:
        source_to_bases[_sample_source(base)].append(base)

    val_bases: set[str] = set()
    for source, bases in source_to_bases.items():
        bases_sorted = sorted(bases)
        rng.shuffle(bases_sorted)
        n = len(bases_sorted)
        if n <= 1:
            continue
        n_val = int(round(n * float(val_ratio)))
        n_val = max(1, min(n - 1, n_val))
        val_bases.update(bases_sorted[:n_val])

    if eliminate_hash_leakage:
        hash_to_bases: dict[str, set[str]] = defaultdict(set)
        for sid in tqdm(mask_stems, desc='Hashing masks for leakage guard'):
            mp = out_dir / 'masks' / f'{sid}.png'
            if not mp.exists():
                continue
            h = _mask_file_hash(mp)
            hash_to_bases[h].add(_sample_base_id(sid))

        changed = True
        while changed:
            changed = False
            train_bases = set(base_to_samples.keys()) - val_bases
            for bases in hash_to_bases.values():
                if len(bases) < 2:
                    continue
                if (bases & train_bases) and (bases & val_bases):
                    # Keep duplicate-signature groups in train to avoid optimistic validation.
                    to_move = bases & val_bases
                    if to_move:
                        val_bases -= to_move
                        changed = True

    train_ids: list[str] = []
    val_ids: list[str] = []
    for base, sids in base_to_samples.items():
        if base in val_bases:
            val_ids.extend(sids)
        else:
            train_ids.extend(sids)

    return sorted(train_ids), sorted(val_ids)


def _write_split_files(out_dir: Path, train_ids: list[str], val_ids: list[str]) -> None:
    (out_dir / 'train.txt').write_text('\n'.join(train_ids) + ('\n' if train_ids else ''), encoding='utf-8')
    (out_dir / 'val.txt').write_text('\n'.join(val_ids) + ('\n' if val_ids else ''), encoding='utf-8')


def _print_split_audit(out_dir: Path, train_ids: list[str], val_ids: list[str]) -> None:
    train_set = set(train_ids)
    val_set = set(val_ids)
    train_bases = {_sample_base_id(x) for x in train_set}
    val_bases = {_sample_base_id(x) for x in val_set}

    print('\n=== Split Audit ===')
    print(f'train={len(train_ids)} val={len(val_ids)} total={len(train_ids) + len(val_ids)}')
    print(f'exact_overlap={len(train_set & val_set)}')
    print(f'base_overlap={len(train_bases & val_bases)}')

    src_train = Counter(_sample_source(x) for x in train_ids)
    src_val = Counter(_sample_source(x) for x in val_ids)
    print(f'source_train={dict(sorted(src_train.items()))}')
    print(f'source_val={dict(sorted(src_val.items()))}')

    # Lightweight leakage check by hash intersection.
    train_hash = set()
    for sid in train_ids:
        mp = out_dir / 'masks' / f'{sid}.png'
        if mp.exists():
            train_hash.add(_mask_file_hash(mp))
    val_hash = set()
    for sid in val_ids:
        mp = out_dir / 'masks' / f'{sid}.png'
        if mp.exists():
            val_hash.add(_mask_file_hash(mp))
    print(f'hash_leakage={len(train_hash & val_hash)}')


def _summarize_masks(out_dir: Path, mask_stems: list[str]) -> None:
    class_px = Counter()
    bad = 0
    for sid in tqdm(mask_stems, desc='Summarizing classes'):
        mp = out_dir / 'masks' / f'{sid}.png'
        m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if m is None:
            bad += 1
            continue
        u, c = np.unique(m, return_counts=True)
        for uu, cc in zip(u.tolist(), c.tolist()):
            class_px[int(uu)] += int(cc)

    total_px = sum(class_px.values())
    pct = {k: round(v * 100.0 / total_px, 3) for k, v in sorted(class_px.items())} if total_px > 0 else {}
    print('\n=== Class Distribution ===')
    print(f'bad_masks={bad}')
    print(f'class_pixels={dict(sorted(class_px.items()))}')
    print(f'class_percent={pct}')


# --- Main ---
def main() -> None:
    args = parse_args()
    out_dir = args.out_dir

    if not args.split_only:
        setup_dirs(out_dir)
        t1 = extract_roof3d(out_dir)
        t2 = extract_rid(out_dir)
        t3 = extract_ninja(out_dir)
        t4 = extract_yolo(out_dir)
        t5 = extract_sodws(out_dir)
        print(f'\nExtraction total saved: {t1 + t2 + t3 + t4 + t5}')

    masks_dir = out_dir / 'masks'
    if not masks_dir.exists():
        raise FileNotFoundError(f'Masks directory not found: {masks_dir}')

    mask_stems = sorted([p.stem for p in masks_dir.glob('*.png')])
    if not mask_stems:
        raise RuntimeError(f'No masks found in {masks_dir}')

    train_ids, val_ids = _build_split(
        out_dir=out_dir,
        mask_stems=mask_stems,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        eliminate_hash_leakage=not bool(args.allow_hash_leakage),
    )

    _write_split_files(out_dir, train_ids, val_ids)
    _print_split_audit(out_dir, train_ids, val_ids)
    _summarize_masks(out_dir, mask_stems)

    print('\n' + '=' * 60)
    print('MASSIVE DATASET V2 DONE')
    print(f'out_dir={out_dir}')
    print(f'train.txt entries={len(train_ids)}')
    print(f'val.txt entries={len(val_ids)}')
    print('=' * 60)


if __name__ == '__main__':
    main()
