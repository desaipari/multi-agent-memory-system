python -c "
import json, os, re

def normalize(s):
    if not s: return ''
    s = str(s).lower().strip()
    s = re.sub(r'[\s\-_]+', '', s)
    return s

scenarios_dir = '../dataset/scenarios'
for filename in sorted(os.listdir(scenarios_dir)):
    if not filename.endswith('.json'): continue
    with open(os.path.join(scenarios_dir, filename), encoding='utf-8') as f:
        data = json.load(f)
    for s in data.get('scenarios', []):
        gt = s.get('ground_truth', {})
        if not gt.get('contradiction_expected'): continue
        correct = normalize(gt.get('correct_resolution', ''))
        if not correct: continue
        turns = s.get('turns', [])
        values = [normalize(t.get('value','')) for t in turns]
        match = correct in values
        if not match:
            print(f\"{s.get('scenario_id')}: correct='{correct}' turn_values={values} MISMATCH\")
        else:
            print(f\"{s.get('scenario_id')}: correct='{correct}' MATCH\")
"