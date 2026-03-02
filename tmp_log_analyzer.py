import json
import os
import sys
from collections import defaultdict

def analyze_json_log(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    print(f"\n--- Analyzing Log: {filepath} ---")
    val_data = []
    train_data = []

    try:
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'mIoU' in data or 'facet/AP50' in data:
                        val_data.append(data)
                    elif 'loss' in data and 'step' in data:
                        train_data.append(data)
                except Exception:
                    pass
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    print(f"Parsed {len(train_data)} training records and {len(val_data)} validation records.")

    print("\n[ Validation Metrics (Top 20) ]")
    for v in val_data[-20:]:  # Print last 20 val records
        step = v.get('step', -1)
        mIoU = v.get('mIoU', 0.0)
        aAcc = v.get('aAcc', 0.0)
        ap50 = v.get('facet/AP50', 0.0)
        print(f"Step: {step:5d} | AP50: {ap50:.2f} | mIoU: {mIoU:.4f} | aAcc: {aAcc:.4f}")

    print("\n[ Training Losses (Every 1000th iteration or last few) ]")
    for t in train_data:
        step = t.get('step', -1)
        if step % 1000 == 0 or step % 5000 == 0 or step == train_data[-1].get('step'):
            print(f"Step: {step:5d} | loss: {t.get('loss', 0):.4f} | cls: {t.get('loss_cls', 0):.4f} | mask: {t.get('loss_mask', 0):.4f} | dice: {t.get('loss_dice', 0):.4f}")

if __name__ == '__main__':
    paths = ['/Users/voskan/Desktop/DeepRoof-2026/20260302_141731.json', '/Users/voskan/Desktop/DeepRoof-2026/scalars.json']
    for p in paths:
        analyze_json_log(p)
