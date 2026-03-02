import json
import os
import pprint

def deep_analyze_logs(filepaths):
    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
            
        print(f"\n{'='*50}\nAnalyzing {filepath}\n{'='*50}")
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                
            metrics_5k = None
            metrics_10k = None
            
            for line in lines:
                try:
                    data = json.loads(line.strip())
                    if 'step' in data and 'facet/AP50' in data:
                        step = data['step']
                        if step == 5000:
                            metrics_5k = data
                        if step == 10000:
                            metrics_10k = data
                except json.JSONDecodeError:
                    pass
            
            print("--- Validation Metrics at 5k ---")
            pprint.pprint(metrics_5k)
            print("--- Validation Metrics at 10k ---")
            pprint.pprint(metrics_10k)

        except Exception as e:
            print(f"Error reading {filepath}: {e}")

if __name__ == "__main__":
    deep_analyze_logs([
        '/Users/voskan/Desktop/DeepRoof-2026/20260302_080317.json'
    ])
