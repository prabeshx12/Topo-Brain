"""Build a POSIX-safe code bundle zip for the Kaggle kernels."""
import zipfile
from pathlib import Path

REPO = Path(r"d:\BCT_pcampus\Semester VII\Major Project\topobrain\Topo-Brain")
OUT = Path(
    r"C:\Users\Asus\AppData\Local\Temp\claude"
    r"\d--BCT-pcampus-Semester-VII-Major-Project-topobrain-Topo-Brain"
    r"\f08ee9d9-7f5a-4048-bba5-59d5196c35a6\scratchpad\kgcode\topobrain_code.zip"
)
OUT.parent.mkdir(parents=True, exist_ok=True)

FILES = {
    "src/model.py": REPO / "src/model.py",
    "src/model_cascaded.py": REPO / "src/model_cascaded.py",
    "src/diffusion.py": REPO / "src/diffusion.py",
    "src/topology_loss.py": REPO / "src/topology_loss.py",
    "src/synthesis_dataset.py": REPO / "src/synthesis_dataset.py",
    "src/metrics_honest.py": REPO / "src/metrics_honest.py",
    "scripts/train_cascaded.py": REPO / "scripts/train_cascaded.py",
    "scripts/evaluate_full_volume.py": REPO / "scripts/evaluate_full_volume.py",
    "scripts/compute_betti.py": REPO / "scripts/compute_betti.py",
    "configs/train_diffusion.yaml": REPO / "configs/train_diffusion.yaml",
}

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for arc, src in FILES.items():
        assert src.exists(), f"missing {src}"
        z.write(src, arcname=arc)
    # minimal inits so `src.model_cascaded` imports without pulling in monai via src/__init__
    z.writestr("src/__init__.py", "# minimal init for Kaggle (avoids monai import)\n")
    z.writestr("scripts/__init__.py", "")

with zipfile.ZipFile(OUT) as z:
    for n in z.namelist():
        print(n)
print("OK ->", OUT, OUT.stat().st_size, "bytes")
