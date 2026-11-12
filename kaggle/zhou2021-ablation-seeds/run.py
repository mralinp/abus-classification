import glob, os, subprocess, sys
import torch
OUT, REPO = "/kaggle/working", "/kaggle/working/repo"
sh = lambda c: subprocess.run(c, shell=True, check=True)
assert torch.cuda.device_count() == 2, f"expected T4 x 2, found {torch.cuda.device_count()} GPUs"
sh(f"git clone -q https://github.com/mralinp/abus-classification.git {REPO} && git -C {REPO} rev-parse HEAD")   # main, commit in the log
sh("pip install -q pynrrd trimesh tdsc-abus2023-pytorch")
cache = glob.glob("/kaggle/input/**/eswa2025.npz", recursive=True)[0]   # features from the eswa2025-reproduction run
os.makedirs(f"{REPO}/data")
os.symlink(cache, f"{REPO}/data/eswa2025.npz")
# The paper's Table 1 models and lambda = 0.7 for seeds 1 and 2, with the settings of the seed-0 kernels (100 epochs),
# one seed per GPU. Each seed writes to its own folder, so the two never share a file.
code = ("import sys; from abus_classification.papers.zhou2021 import ablations; "
        "ablations(100, seed=int(sys.argv[1]), ckpt_dir=sys.argv[2])")
jobs = []
for gpu, seed in enumerate((1, 2)):
    os.makedirs(f"{OUT}/seed{seed}", exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": REPO}
    jobs.append(subprocess.Popen([sys.executable, "-c", code, str(seed), f"{OUT}/seed{seed}"], env=env, cwd=REPO))
codes = [j.wait() for j in jobs]
sh(f"rm -rf {REPO}")
assert not any(codes), f"exit codes {codes}"
