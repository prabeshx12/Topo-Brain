import numpy as np

subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05',
            'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']

rng = np.random.default_rng(42)
rng.shuffle(subjects)

print("Fold assignments:")
for i, s in enumerate(subjects):
    label = "VAL" if i == 0 else "TEST" if i == 1 else "train"
    print(f"  Fold {i}: {s}  [{label}]")
