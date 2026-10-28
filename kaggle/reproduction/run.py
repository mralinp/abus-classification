import os, subprocess, sys
sh = lambda c: subprocess.run(c, shell=True, check=True)
sh("git clone -q -b feat/eswa2025-kaggle https://github.com/mralinp/abus-classification.git /kaggle/working/repo")
sh("pip install -q pynrrd trimesh gdown tdsc-abus2023-pytorch")

# Datasets are extracted and nested differently (Train/Train/DATA vs Test/DATA): link each split folder into /tmp/tdsc.
os.makedirs("/tmp/tdsc", exist_ok=True)
for root, dirs, _ in os.walk("/kaggle/input"):
    if "DATA" in dirs and os.path.basename(root) in ("Train", "Validation", "Test"):
        os.symlink(root, f"/tmp/tdsc/{os.path.basename(root)}")
print(os.listdir("/tmp/tdsc"), flush=True)
assert len(os.listdir("/tmp/tdsc")) == 3, "missing a split"

os.environ["TDSC_PATH"] = "/tmp/tdsc"
sys.path.insert(0, "/kaggle/working/repo")
import reproduce_eswa2025 as r
r.main(epochs=60).to_csv("/kaggle/working/results.csv")
sh("cp /kaggle/working/repo/data/eswa2025.npz /kaggle/working/")   # keep the cache as kernel output
