import json

# Fix training notebook
nb_path = '/Users/voskan/Desktop/DeepRoof-2026/notebooks/train_deeproof.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

patched = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    for i, line in enumerate(cell['source']):
        # Fix 1: checkpoint interval should be 5000 not 500
        if "cfg.default_hooks.checkpoint.interval = 500" in line:
            cell['source'][i] = "cfg.default_hooks.checkpoint.interval = 5000\n"
            patched += 1
            print(f"  Fixed: checkpoint.interval 500 → 5000")

        # Fix 2: val_interval should NOT be forced to 500
        if "cfg.train_cfg.val_interval = min(int(cfg.train_cfg.get('val_interval', 5000)), 500)" in line:
            cell['source'][i] = "    pass  # val_interval already set in config (5000)\n"
            patched += 1
            print(f"  Fixed: removed val_interval override to 500")

        # Fix 3: NOTEBOOK_NUM_POINTS should match config (12544)  
        if "NOTEBOOK_NUM_POINTS = 2048" in line:
            cell['source'][i] = "NOTEBOOK_NUM_POINTS = 12544\n"
            patched += 1
            print(f"  Fixed: NOTEBOOK_NUM_POINTS 2048 → 12544")

        # Fix 4: max_keep_ckpts from 5 to 3 (match config)
        if "cfg.default_hooks.checkpoint.max_keep_ckpts = 5" in line:
            cell['source'][i] = "cfg.default_hooks.checkpoint.max_keep_ckpts = 3\n"
            patched += 1
            print(f"  Fixed: max_keep_ckpts 5 → 3")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"\nTotal patches applied: {patched}")
