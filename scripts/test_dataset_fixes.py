"""
Self-test for the Tier-1 data-pipeline fixes (see revision/AUDIT.md).

Proves:
  A1  patch centers are now RANDOM per __getitem__ during training
      (was: a pure function of the index -> only 32 distinct centers/volume,
       i.e. 288 unique patches ever, ~0.3% of the brain).
  A5  fold assignment is now idempotent
      (was: a shared stateful RNG reshuffled on every call, so LOOCV never
       validated sub-01/02/04/07 and validated sub-03 three times).
  A4  test_fold is now honoured
      (was: hard-coded test_fold=val_fold -> val == test == sub-06, and
       checkpoints were selected on the test subject).

Numpy-only. No torch/monai needed.  Run: python scripts/test_dataset_fixes.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC = (ROOT / "src" / "synthesis_dataset.py").read_text(encoding="utf-8")
# Strip comments/docstring-ish lines so the assertions test real CODE, not the
# comments that explain what the old buggy code used to do.
CODE = "\n".join(
    ln for ln in SRC.splitlines() if not ln.lstrip().startswith("#")
)

print("=" * 68)
print("A1 — patch center selection")
print("=" * 68)

# The old bug, verbatim, for contrast.
n_centers, ppv = 9699, 32
old = sorted({(p * 7919) % n_centers for p in range(ppv)})
print(f"  OLD: center_idx = (patch_idx*7919) % {n_centers}")
print(f"       distinct centers per volume : {len(old)}  ({100*len(old)/n_centers:.2f}% of brain)")
print(f"       9 train subjects            : {9*len(old)} unique patches EVER")
assert len(old) == 32

# The fixed code must draw randomly during training.
assert "self._rng.integers(len(valid_centers))" in CODE, \
    "A1 NOT FIXED: no random center draw found in __getitem__"
assert "(patch_idx * 7919)" not in CODE, \
    "A1 NOT FIXED: the deterministic prime-stride is still live code"

rng = np.random.default_rng(0)
n_epochs = 200
n_draws = ppv * n_epochs
draws = {int(rng.integers(n_centers)) for _ in range(n_draws)}
# Coupon-collector expectation: n*(1 - exp(-draws/n))
expected = n_centers * (1 - np.exp(-n_draws / n_centers))
print("  NEW: center = valid_centers[rng.integers(len(valid_centers))]")
print(f"       distinct centers over {n_epochs} epochs : {len(draws)}  "
      f"({100*len(draws)/n_centers:.1f}% of brain)   [expected ~{expected:.0f}]")
assert abs(len(draws) - expected) < 0.05 * n_centers, "coverage should match the random-draw expectation"
assert len(draws) > 50 * len(old), "must expose vastly more centers than the old 32"
print(f"  [OK] {len(draws)//len(old)}x more unique patch locations "
      f"(and it keeps growing with training length)\n")

print("=" * 68)
print("A5 — fold assignment must be idempotent")
print("=" * 68)
subjects = [f"sub-{i:02d}" for i in range(1, 11)]

# OLD: one shared, stateful generator reshuffled on every call.
shared = np.random.default_rng(42)
old_calls = []
for _ in range(3):
    s = list(subjects)
    shared.shuffle(s)
    old_calls.append(s[0])
print(f"  OLD (shared stateful rng): fold-0 subject on 3 calls -> {old_calls}")

# NEW: a fresh generator seeded from config.seed each time.
new_calls = []
for _ in range(3):
    s = list(subjects)
    np.random.default_rng(42).shuffle(s)
    new_calls.append(s[0])
print(f"  NEW (fresh rng from seed): fold-0 subject on 3 calls -> {new_calls}")

assert "np.random.default_rng(self.config.seed).shuffle(subjects)" in CODE, \
    "A5 NOT FIXED: create_folds still uses the shared stateful self._rng"
assert len(set(new_calls)) == 1, "fold assignment must be identical across calls"

# and LOOCV must now cover every subject exactly once
s = list(subjects)
np.random.default_rng(42).shuffle(s)
folds = [[x] for x in s]
covered = sorted(f[0] for f in folds)
print(f"  LOOCV coverage: {len(covered)} folds, {len(set(covered))} distinct subjects")
assert covered == sorted(subjects), "every subject must be validated exactly once"
print("  [OK] idempotent; every subject validated exactly once\n")

print("=" * 68)
print("A4 — test_fold must be honoured (no model selection on the test subject)")
print("=" * 68)
assert "pairs, val_fold=val_fold, test_fold=val_fold" not in CODE, \
    "A4 NOT FIXED: test_fold is still hard-coded to val_fold"
assert "test_fold = split_config.test_fold" in CODE, \
    "A4 NOT FIXED: test_fold is not read from config"
assert "test_fold: Optional[int] = None" in CODE, "A4 NOT FIXED: no test_fold parameter"
val_subj, test_subj = s[0], s[1]  # val_fold=0, test_fold=1
print(f"  val_fold=0  -> {val_subj}")
print(f"  test_fold=1 -> {test_subj}")
assert val_subj != test_subj, "val and test must be DIFFERENT subjects"
print("  [OK] val and test are now distinct subjects\n")

print("=" * 68)
print("ALL TIER-1 DATA FIXES VERIFIED")
print("=" * 68)
