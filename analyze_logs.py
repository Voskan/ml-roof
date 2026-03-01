import json

log_file = '20260228_071732.json'
scalars_file = 'scalars.json'

val_metrics = []
train_losses = []

try:
    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'mIoU' in data:
                    val_metrics.append({'step': data.get('step'), 'mIoU': data['mIoU'], 'aAcc': data.get('aAcc')})
                if 'loss' in data and data.get('step') % 1000 == 0:
                    train_losses.append({'step': data.get('step'), 'loss': data['loss'], 'loss_cls': data.get('loss_cls'), 'loss_mask': data.get('loss_mask'), 'lr': data.get('lr')})
            except Exception:
                pass
except Exception as e:
    print(f"Error reading {log_file}: {e}")

try:
    with open(scalars_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'mIoU' in data:
                    val_metrics.append({'step': data.get('step'), 'mIoU': data['mIoU'], 'aAcc': data.get('aAcc')})
                if 'loss' in data and data.get('step') and data.get('step') % 5000 == 0:
                    train_losses.append({'step': data.get('step'), 'loss': data['loss'], 'loss_cls': data.get('loss_cls'), 'loss_mask': data.get('loss_mask'), 'lr': data.get('lr')})
            except Exception:
                pass
except Exception as e:
    print(f"Error reading {scalars_file}: {e}")

# Deduplicate and sort
val_metrics = sorted([dict(t) for t in {tuple(d.items()) for d in val_metrics}], key=lambda x: x['step'] if x['step'] else 0)
train_losses = sorted([dict(t) for t in {tuple(d.items()) for d in train_losses}], key=lambda x: x['step'] if x['step'] else 0)

print("\n--- VALIDATION METRICS ---")
for v in val_metrics:
    print(f"Step {v['step']}: mIoU = {v['mIoU']:.4f}, aAcc = {v.get('aAcc', 0):.4f}")

print("\n--- TRAINING LOSSES (every 5k steps) ---")
for t in train_losses:
    if t['step'] and t['step'] % 5000 == 0:
        print(f"Step {t['step']}: loss = {t['loss']:.4f}, loss_cls = {t['loss_cls']:.4f}, loss_mask = {t['loss_mask']:.4f}, lr = {t['lr']:.2e}")

