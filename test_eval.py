import sys
from pathlib import Path

project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.utils import register_all_modules

register_all_modules(init_default_scope=False)

import deeproof.models.backbones.swin_v2_compat
import deeproof.models.deeproof_model
import deeproof.models.heads.mask2former_head
import deeproof.models.heads.geometry_head
import deeproof.models.losses
import deeproof.datasets.roof_dataset

from deeproof.utils.runtime_compat import apply_runtime_compat

cfg = Config.fromfile('configs/deeproof_production_swin_L.py')
cfg.default_scope = 'mmseg'
cfg.work_dir = 'work_dirs/test_eval'

cfg.data_root = str(project_root / 'data/OmniCity/')
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_root = cfg.data_root

if cfg.get('val_cfg') is None:
    cfg.val_cfg = dict(type='ValLoop')

apply_runtime_compat(cfg)

# Only run validation to see if it crashes and what metrics it outputs
runner = Runner.from_cfg(cfg)
print("Starting validation...")
try:
    runner.val_loop.run()
    print("Validation successful")
except Exception as e:
    import traceback
    traceback.print_exc()
