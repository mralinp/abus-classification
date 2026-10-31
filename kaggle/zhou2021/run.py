import glob, os, subprocess, sys
sh = lambda c: subprocess.run(c, shell=True, check=True)
sh("git clone -q -b feat/zhou2021 https://github.com/mralinp/abus-classification.git /kaggle/working/repo")
sh("pip install -q pynrrd trimesh tdsc-abus2023-pytorch")
cache = glob.glob("/kaggle/input/**/eswa2025.npz", recursive=True)[0]   # features from the eswa2025-reproduction run
os.makedirs("/kaggle/working/repo/data")
os.symlink(cache, "/kaggle/working/repo/data/eswa2025.npz")
sys.path.insert(0, "/kaggle/working/repo")
# Resume from checkpoints of an earlier (crashed) run: attach it as a kernel source, e.g. "mralinp/zhou2021-multitask".
for f in glob.glob("/kaggle/input/**/zhou2021_seed*.pt", recursive=True):
    sh(f"cp -n {f} /kaggle/working/")
from abus_classification.papers.zhou2021 import main
# Outputs per seed: checkpoint, test probabilities, <seed>_history.csv (focal / Dice terms, train AUC per epoch) and
# <seed>_test.npz (per-case Dice, mid-plane probability maps). Finished checkpoints attached as input are only
# re-evaluated, which gives the test npz without retraining. val=True: diagnostic run with 20% of Train+Validation held out.
main(100, ckpt_dir="/kaggle/working", val=False).to_csv("/kaggle/working/zhou2021_results.csv")
