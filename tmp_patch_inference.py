import json

# Fix inference notebook
nb_path = '/Users/voskan/Desktop/DeepRoof-2026/notebooks/checkpoint_inference_test.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

patched = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    for i, line in enumerate(cell['source']):
        # Fix: Default checkpoint should prefer best, not iter_500
        if "CHECKPOINT_NAME = 'iter_500.pth'" in line:
            cell['source'][i] = "CHECKPOINT_NAME = 'best_mIoU.pth'  # Prefer best model\n"
            patched += 1
            print(f"  Fixed: CHECKPOINT_NAME iter_500 → best_mIoU")

        # Fix: TILE_SIZE=1408 is too large for 512px images, use 512
        if "TILE_SIZE = 1408" in line:
            cell['source'][i] = "TILE_SIZE = 512\n"
            patched += 1
            print(f"  Fixed: TILE_SIZE 1408 → 512")

        # Fix: STRIDE=1024 is too large, use 384 for overlap
        if "STRIDE = 1024" in line:
            cell['source'][i] = "STRIDE = 384\n"
            patched += 1
            print(f"  Fixed: STRIDE 1024 → 384")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"\nTotal patches applied: {patched}")
