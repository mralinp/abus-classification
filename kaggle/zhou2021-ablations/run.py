import glob, os, subprocess, sys
sh = lambda c: subprocess.run(c, shell=True, check=True)
sh("git clone -q -b feat/zhou2021 https://github.com/mralinp/abus-classification.git /kaggle/working/repo")
sh("pip install -q pynrrd trimesh tdsc-abus2023-pytorch")
cache = glob.glob("/kaggle/input/**/eswa2025.npz", recursive=True)[0]   # features from the eswa2025-reproduction run
os.makedirs("/kaggle/working/repo/data")
os.symlink(cache, "/kaggle/working/repo/data/eswa2025.npz")
sys.path.insert(0, "/kaggle/working/repo")
# Seed 0: the paper's method with 20% of Train+Validation held out (focal / Dice terms, train and val AUC per epoch),
# then the ablations. All files have their own names; zhou2021-multitask is not an input here.
from abus_classification.papers.zhou2021 import ablations, main
main(100, seeds=(0,), ckpt_dir="/kaggle/working", val=True).to_csv("/kaggle/working/zhou2021_val_results.csv")
ablations(100, ckpt_dir="/kaggle/working")
sh("rm -rf /kaggle/working/repo")
