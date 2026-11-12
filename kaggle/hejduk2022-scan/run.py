"""Test scan of Hejduk 2022 with the weights trained by the kernel `hejduk2022-early-stopping` (no training here).

Window stride 20 on every fifth slice (paper: stride 10, every slice, about 20 times the work and days per seed on a T4).
One seed per GPU. Each scanned volume is saved, and a later run that has this
kernel's output as an input continues from those files.
"""
import glob, os, shutil, subprocess, sys
OUT, REPO = "/kaggle/working", "/kaggle/working/repo"
sh = lambda c: subprocess.run(c, shell=True, check=True)

if __name__ == "__main__":
    import torch
    assert torch.cuda.device_count() == 2, f"expected T4 x 2, found {torch.cuda.device_count()} GPUs"
    sh(f"git clone -q -b feat/hejduk2022-early-stopping https://github.com/mralinp/abus-classification.git {REPO} && git -C {REPO} rev-parse HEAD")
    sh("pip install -q pynrrd trimesh tdsc-abus2023-pytorch")
    os.makedirs("/tmp/tdsc", exist_ok=True)
    for root, dirs, _ in os.walk("/kaggle/input"):
        if "DATA" in dirs and os.path.basename(root) in ("Train", "Validation", "Test"):
            os.symlink(root, f"/tmp/tdsc/{os.path.basename(root)}")
    assert len(os.listdir("/tmp/tdsc")) == 3, "missing a split"
    os.makedirs(f"{REPO}/data", exist_ok=True)
    os.symlink(glob.glob("/kaggle/input/**/hejduk2022.npz", recursive=True)[0], f"{REPO}/data/hejduk2022.npz")
    weights = glob.glob("/kaggle/input/**/hejduk2022_model*_es_lr0.001_seed*.pt", recursive=True)
    assert len(weights) == 9, f"expected 9 trained models, found {len(weights)}"
    for path in weights:                                              # links, so 2.6 GB of weights are not copied into this output
        os.symlink(path, f"{OUT}/{os.path.basename(path)}")
    for path in glob.glob("/kaggle/input/**/hejduk2022_es_lr0.001_seed*_test*.npz", recursive=True):   # volumes an earlier scan finished
        shutil.copy(path, OUT)
    os.environ["TDSC_PATH"] = "/tmp/tdsc"
    sys.path.insert(0, REPO)
    from abus_classification.papers.hejduk2022 import main
    main(seeds=(0, 1, 2), out_dir=OUT, lr=1e-3, stride=20, slice_step=5, hours=11)   # finished models are loaded, not retrained
    for path in glob.glob(f"{OUT}/*.pt"):
        os.unlink(path)
    shutil.rmtree(REPO)
