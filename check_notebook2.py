import json

notebook_path = 'notebooks/train_deeproof.ipynb'
try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        for i, cell in enumerate(nb.get('cells', [])):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                if 'AUTO_BATCH_SIZE' in source or 'BATCH_SIZE' in source.upper():
                    print(f"--- Cell {i} (contains auto batch size) ---")
                    print(source[:500])
                    print("...\n")
except Exception as e:
    print(f"Error reading notebook: {e}")

