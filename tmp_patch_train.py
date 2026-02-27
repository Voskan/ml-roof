import json

notebook_path = '/Users/voskan/Desktop/DeepRoof-2026/notebooks/train_deeproof.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

patched = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            # Fix DATA_ROOT
            if "DATA_ROOT = project_root / 'data'" in line and 'MassiveMasterDataset' not in line:
                cell['source'][i] = "DATA_ROOT = project_root / 'data' / 'MassiveMasterDataset'\n"
                patched += 1
            # Fix NOTEBOOK_IMAGE_SIZE
            if 'NOTEBOOK_IMAGE_SIZE' in line and '1024' in line:
                cell['source'][i] = "NOTEBOOK_IMAGE_SIZE = (512, 512)\n"
                patched += 1
            # Fix NOTEBOOK_NUM_QUERIES
            if 'NOTEBOOK_NUM_QUERIES' in line and '128' in line:
                cell['source'][i] = "NOTEBOOK_NUM_QUERIES = 100\n"
                patched += 1

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Patched {patched} lines in train notebook.")
