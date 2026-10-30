import glob, os, subprocess, sys
sh = lambda c: subprocess.run(c, shell=True, check=True)
sh("git clone -q -b feat/zhou2021 https://github.com/mralinp/abus-classification.git /kaggle/working/repo")
sh("pip install -q pynrrd trimesh tdsc-abus2023-pytorch")
cache = glob.glob("/kaggle/input/**/eswa2025.npz", recursive=True)[0]   # features from the eswa2025-reproduction run
os.makedirs("/kaggle/working/repo/data")
os.symlink(cache, "/kaggle/working/repo/data/eswa2025.npz")
sys.path.insert(0, "/kaggle/working/repo")
# Evaluate the finished zhou2021-multitask checkpoints (read-only input): per-case Dice and mid-plane maps, no training.
for f in glob.glob("/kaggle/input/**/zhou2021_seed?.pt", recursive=True):
    sh(f"cp {f} /kaggle/working/")
from abus_classification.papers.zhou2021 import main
main(100, ckpt_dir="/kaggle/working").to_csv("/kaggle/working/zhou2021_results.csv")
sh("rm /kaggle/working/zhou2021_seed?.pt")                               # copies; the originals stay in the input kernel
sh("rm -rf /kaggle/working/repo")
