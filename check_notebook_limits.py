import json

notebook_path = 'notebooks/train_deeproof.ipynb'
try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        for i, cell in enumerate(nb.get('cells', [])):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                if 'NOTEBOOK_BATCH_SIZE' in source:
                    print(f"--- Cell {i} ---")
                    lines = [ln for ln in source.split('\n') if 'BATCH_SIZE' in ln]
                    print('\n'.join(lines))
except Exception as e:
    print(f"Error reading notebook: {e}")

