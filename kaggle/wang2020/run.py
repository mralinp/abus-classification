import os, subprocess, sys
sh = lambda c: subprocess.run(c, shell=True, check=True)
sh("git clone -q -b feat/wang2020 https://github.com/mralinp/abus-classification.git /kaggle/working/repo")
sh("pip install -q pynrrd trimesh tdsc-abus2023-pytorch")
os.makedirs("/tmp/tdsc", exist_ok=True)
for root, dirs, _ in os.walk("/kaggle/input"):
    if "DATA" in dirs and os.path.basename(root) in ("Train", "Validation", "Test"):
        os.symlink(root, f"/tmp/tdsc/{os.path.basename(root)}")
assert len(os.listdir("/tmp/tdsc")) == 3, "missing a split"
os.environ["TDSC_PATH"] = "/tmp/tdsc"
sys.path.insert(0, "/kaggle/working/repo")
from abus_classification.papers.wang2020 import main
main(out_dir="/kaggle/working")
sh("cp /kaggle/working/repo/data/wang2020.npz /kaggle/working/")
sh("rm -rf /kaggle/working/repo")
