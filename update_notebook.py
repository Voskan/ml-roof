import json

notebook_path = 'notebooks/train_deeproof.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        for i, line in enumerate(source):
            if line.startswith('NOTEBOOK_BATCH_SIZE ='):
                source[i] = "NOTEBOOK_BATCH_SIZE = 22\n"
            if line.startswith('NOTEBOOK_BASELINE_BATCH_SIZE ='):
                source[i] = "NOTEBOOK_BASELINE_BATCH_SIZE = 22\n"
            if line.startswith('NOTEBOOK_MAX_BATCH_SIZE ='):
                source[i] = "NOTEBOOK_MAX_BATCH_SIZE = 24\n"
            if line.startswith('NOTEBOOK_AUTO_TUNE_BATCH ='):
                source[i] = "NOTEBOOK_AUTO_TUNE_BATCH = False\n"

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
