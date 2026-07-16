import importlib.util
from pathlib import Path
import torch
R = Path(r"d:\BCT_pcampus\Semester VII\Major Project\topobrain\Topo-Brain")
spec = importlib.util.spec_from_file_location("te", R/"src"/"topology_euler.py")
te = importlib.util.module_from_spec(spec); spec.loader.exec_module(te)

S = 13
tgt = torch.zeros(1, S, S, S, dtype=torch.long); tgt[0, 3:10, 3:10, 3:10] = 1
loss_fn = te.EulerTopologyLoss(num_classes=2, include_background=False)

def logits_for(mask, conf):
    lg = torch.zeros(1, 2, S, S, S); lg[0, 1] = mask*(2*conf)-conf; lg[0, 0] = -lg[0, 1]; return lg

solid = torch.zeros(S, S, S); solid[3:10, 3:10, 3:10] = 1
holed = solid.clone(); holed[:, 6, 6] = 0            # broken topology (a tunnel)

print("The confound: soft loss falls with CONFIDENCE even when topology never changes.")
print("The monitor (hard chi) must ignore confidence and track only the DISCRETE topology.\n")
print("scenario                         soft_loss   topo_monitor(hard)")
print("-"*66)
for c in (1.0, 3.0, 8.0):
    o = loss_fn(logits_for(solid, c), tgt)
    print(f"CORRECT shape, conf={c:<4}          {float(o['loss']):8.4f}   {float(o['topo_monitor']):8.4f}")
for c in (1.0, 3.0, 8.0):
    o = loss_fn(logits_for(holed, c), tgt)
    print(f"BROKEN shape,  conf={c:<4}          {float(o['loss']):8.4f}   {float(o['topo_monitor']):8.4f}")

print("\nEXPECT: soft_loss falls with confidence in BOTH blocks (the confound).")
print("        topo_monitor = 0 for CORRECT at all confidences, and STAYS >0 for BROKEN.")
so = loss_fn(logits_for(solid, 8.0), tgt); ho = loss_fn(logits_for(holed, 8.0), tgt)
assert float(so['topo_monitor']) < 1e-6, "monitor must be 0 for a topologically-correct shape"
assert float(ho['topo_monitor']) > 0.4, "monitor must stay high for a broken shape regardless of confidence"
# and it must be INVARIANT to confidence on the broken shape (the whole point)
lo = float(loss_fn(logits_for(holed, 1.0), tgt)['topo_monitor'])
hi = float(loss_fn(logits_for(holed, 8.0), tgt)['topo_monitor'])
assert abs(hi - lo) < 1e-3, f"monitor must not change with confidence: {lo} vs {hi}"
print("\n[OK] the monitor reflects TRUE topology, not softmax confidence.")
