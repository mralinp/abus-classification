import glob, os, subprocess, sys
sh = lambda c: subprocess.run(c, shell=True, check=True)
sh("git clone -q -b feat/eswa2025-kaggle https://github.com/mralinp/abus-classification.git /kaggle/working/repo")
sh("pip install -q pynrrd trimesh tdsc-abus2023-pytorch")
cache = glob.glob("/kaggle/input/**/eswa2025.npz", recursive=True)[0]   # features from the eswa2025-reproduction run
os.makedirs("/kaggle/working/repo/data")
os.symlink(cache, "/kaggle/working/repo/data/eswa2025.npz")
sys.path.insert(0, "/kaggle/working/repo")
import reproduce_eswa2025 as r
for s in (1, 2):
    r.main(60, s).to_csv(f"/kaggle/working/results_seed{s}.csv")
