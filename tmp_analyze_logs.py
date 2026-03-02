import json
import sys

def analyze_json_log(filepath):
    print(f"--- Analyzing {filepath} ---")
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        metrics_5k = None
        metrics_15k = None
        metrics_best = None
        best_ap50 = -1
        
        for line in lines:
            try:
                data = json.loads(line.strip())
                if 'step' in data and 'facet/AP50' in data:
                    step = data['step']
                    ap50 = data['facet/AP50']
                    
                    if step == 5000:
                        metrics_5k = data
                    if step == 15000:
                        metrics_15k = data
                        
                    if ap50 > best_ap50:
                        best_ap50 = ap50
                        metrics_best = data
                        
            except json.JSONDecodeError:
                pass
                
        print(f"Metrics at 5k: {metrics_5k}")
        print(f"Metrics at 15k: {metrics_15k}")
        print(f"Best metrics: {metrics_best}")
        
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

if __name__ == "__main__":
    analyze_json_log('/Users/voskan/Desktop/DeepRoof-2026/scalars.json')
    analyze_json_log('/Users/voskan/Desktop/DeepRoof-2026/20260301_133401.json')
