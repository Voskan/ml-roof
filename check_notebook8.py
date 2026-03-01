import json

notebook_path = 'notebooks/train_deeproof.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)
    print("--- Cell 5 Full Content ---")
    print(''.join(nb.get('cells', [])[5].get('source', [])))
