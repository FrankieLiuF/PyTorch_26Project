"""Check whether each dataset is balanced or imbalanced"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent

# One-time utility: compute class balance ratio for each preprocessed dataset
DATASETS = {
    'iris':                DATA_DIR / '1.Numerical_iris/iris.csv',
    'parkinsons':          DATA_DIR / '2.Numerical_parkinsons/parkinsons.csv',
    'hepatitis':           DATA_DIR / '3.Mixed_hepatitis/hepatitis.csv',
    'acute_inflammations': DATA_DIR / '4.Mixed_acute_inflammations/acute_inflammations.csv',
    'zoo':                 DATA_DIR / '5.Categorical_zoo/zoo.csv',
    'hayes_roth':          DATA_DIR / '6.Categorical_hayes_roth/hayes_roth.csv',
    #'lenses':              DATA_DIR / '6.Categorical_lenses/lenses.csv',    
}

rows = []

for name, path in DATASETS.items():
    df = pd.read_csv(path)

    # Count how many samples per class: {0:30, 1:70}
    counts = df['target'].value_counts().sort_index()

    # Connect with '/' : 30/70
    distribution = '/'.join(str(c) for c in counts.values)

    # Balanced = minority class is at least 70% of the majority class
    minority = counts.min()
    majority = counts.max()
    ratio = minority / majority
    balanced = 'Y' if ratio >= 0.7 else 'N'

    # dataset name (width: 25 strings), {0:50, 1:50}
    print(f'{name:25s} {dict(counts)} ratio={ratio:.2f} balanced={balanced}')

    rows.append({
        'dataset': name,
        'balanced': balanced,
        'class_distribution': distribution,
    })

# Save result
out = DATA_DIR / 'dataset_balance.csv'
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nSaved to {out}")