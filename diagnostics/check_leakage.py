import json
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "datasets" / "PropertyTax_model_ready.csv"
OUT = BASE / "models" / "dataset_debug.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

def analyze():
    df = pd.read_csv(DATA)
    if 'is_late_payment' not in df.columns:
        print('No target column is_late_payment found')
        return
    target = 'is_late_payment'
    X = df.drop(columns=[target])
    y = df[target]

    results = {
        'n_rows': len(df),
        'n_unique_target': int(y.nunique()),
        'columns': {},
    }

    # exact equality
    for c in X.columns:
        try:
            eq = X[c].equals(y)
        except Exception:
            eq = False
        results['columns'][c] = {'exact_equal_to_target': bool(eq)}

    # perfect mapping
    perfect_map = []
    for c in X.columns:
        try:
            groups = df.groupby(c)[target].nunique()
            if len(groups) > 0 and groups.max() == 1:
                perfect_map.append(c)
                results['columns'][c]['perfect_mapping'] = True
            else:
                results['columns'][c]['perfect_mapping'] = False
        except Exception as e:
            results['columns'][c]['perfect_mapping'] = False

    # duplicates (feature-wise)
    dup_rows = df.duplicated(subset=X.columns).sum()
    results['duplicate_rows'] = int(dup_rows)

    # check overlap between random train/test split
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    merged = Xte.merge(Xtr, how='inner')
    results['train_test_row_overlap'] = int(len(merged))

    results['perfect_mapping_columns'] = perfect_map

    with OUT.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print('Wrote', OUT)

if __name__ == '__main__':
    analyze()
