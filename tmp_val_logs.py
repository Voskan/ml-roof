import json
import os

def extract_val_logs(filepaths):
    for filepath in filepaths:
        if not os.path.exists(filepath):
            continue
            
        print(f"\n--- Val logs for {filepath} ---")
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if 'aAcc' in line or 'mIoU' in line or 'AP50' in line:
                    print(line.strip())
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    extract_val_logs([
        '/Users/voskan/Desktop/DeepRoof-2026/20260302_080317.json',
        '/Users/voskan/Desktop/DeepRoof-2026/scalars.json'
    ])
