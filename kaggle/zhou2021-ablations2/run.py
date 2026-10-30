import glob, os, subprocess, sys
sh = lambda c: subprocess.run(c, shell=True, check=True)
sh("git clone -q -b feat/zhou2021 https://github.com/mralinp/abus-classification.git /kaggle/working/repo")
sh("pip install -q pynrrd trimesh tdsc-abus2023-pytorch")
cache = glob.glob("/kaggle/input/**/eswa2025.npz", recursive=True)[0]   # features from the eswa2025-reproduction run
os.makedirs("/kaggle/working/repo/data")
os.symlink(cache, "/kaggle/working/repo/data/eswa2025.npz")
sys.path.insert(0, "/kaggle/working/repo")
# Table 1 models missing from zhou2021-ablations: the encoder-only ClsNet and CVNetIter (seed 0).
from abus_classification.papers.zhou2021 import ablations
ablations(100, ckpt_dir="/kaggle/working", names=("clsnet", "cvnetiter"))
sh("rm -rf /kaggle/working/repo")
