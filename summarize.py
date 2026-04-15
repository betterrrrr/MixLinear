import re
from collections import defaultdict

models = ['PatchTST', 'iTransformer', 'MixLinear', 'MixDLinear']
datasets = ['ETTh1', 'ETTm1', 'ETTm2']
in_lens = ['96', '360', '720']
out_lens = ['96', '192']

data = defaultdict(lambda: defaultdict(dict))

with open('result.txt', 'r') as f:
    lines = f.readlines()

for i in range(len(lines) - 1):
    header = lines[i].strip()
    metrics = lines[i+1].strip()
    
    if header == '' or not header.split('_')[0] in datasets:
        continue
        
    ds = header.split('_')[0]
    
    # Try to parse properties
    # Name format usually: Dataset_seqlen_predlen_Model_...
    parts = header.split('_')
    
    if len(parts) >= 4:
        # e.g., ETTh1_360_96_MixDLinear_ETTh1_ftM_sl360_pl96_...
        sl = parts[1]
        pl = parts[2]
        model = parts[3]
        
        # sometimes format might be Dataset_Model_sl...
        
        if model not in models:
            # Let's search inside the whole string to be safe
            found_model = None
            for m in models:
                if m in header:
                    found_model = m
                    break
            if found_model is None:
                continue
            model = found_model
            
            # find sl and pl using regex
            sl_match = re.search(r'_sl(\d+)_', header)
            pl_match = re.search(r'_pl(\d+)_', header)
            if sl_match:
                sl = sl_match.group(1)
            if pl_match:
                pl = pl_match.group(1)
                
        if ds in datasets and model in models and sl in in_lens and pl in out_lens:
            # Extract mse and mae
            m = re.search(r'mse:([0-9.]+), mae:([0-9.]+)', metrics)
            if m:
                # Store best (lowest mse) in case of multiple seeds/configs
                mse = float(m.group(1))
                mae = float(m.group(2))
                key = f"{sl}_{pl}"
                if key not in data[ds][model] or mse < data[ds][model][key]['mse']:
                    data[ds][model][key] = {'mse': mse, 'mae': mae}

# print Markdown table
for ds in datasets:
    print(f"### Dataset: {ds}")
    print("| Model | " + " | ".join([f"{sl}->{pl}" for sl in in_lens for pl in out_lens]) + " |")
    print("|---" + "|---" * (len(in_lens) * len(out_lens)) + "|")
    
    for model in models:
        row = f"| {model} |"
        for sl in in_lens:
            for pl in out_lens:
                key = f"{sl}_{pl}"
                if key in data[ds][model]:
                    row += f" {data[ds][model][key]['mse']:.4f}/{data[ds][model][key]['mae']:.4f} |"
                else:
                    row += " - |"
        print(row)
    print()
