import os
import sys
import torch

from deeproof.datasets.roof_dataset import DeepRoofDataset

def test_dataset_bg():
    print("Testing DeepRoofDataset background instance extraction...")
    
    # We will instantiate the dataset from the training configuration
    from mmengine.config import Config
    cfg = Config.fromfile('configs/deeproof_production_swin_L.py')
    
    # Setup dataset
    dataset_cfg = cfg.train_dataloader.dataset
    dataset_cfg['test_mode'] = True
    
    try:
        dataset = DeepRoofDataset(**dataset_cfg)
        print(f"Loaded dataset with {len(dataset)} samples.")
        
        # Test a few samples
        for i in range(min(5, len(dataset))):
            data = dataset[i]
            data_sample = data['data_samples']
            gt_instances = data_sample.gt_instances
            
            labels = gt_instances.labels
            print(f"Sample {i} Labels: {labels.tolist()}")
            if 0 in labels.tolist():
                print(f"  -> SUCCESS: Background instance (0) found in sample {i}!")
            else:
                print(f"  -> WARNING: Background instance (0) missing in sample {i}.")
                
    except Exception as e:
        print(f"Failed to initialize dataset: {e}")

if __name__ == '__main__':
    test_dataset_bg()
