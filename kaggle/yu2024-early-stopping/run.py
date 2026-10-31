import os, shutil, subprocess, sys
OUT = "/kaggle/working"
sh = lambda c: subprocess.run(c, shell=True, check=True)


def setup():
    sh(f"git clone -q -b feat/yu2024-early-stopping https://github.com/mralinp/abus-classification.git {OUT}/repo")
    sh("pip install -q pynrrd trimesh tdsc-abus2023-pytorch")
    os.makedirs("/tmp/tdsc", exist_ok=True), os.makedirs(f"{OUT}/repo/data", exist_ok=True)
    for root, dirs, files in os.walk("/kaggle/input"):
        if "DATA" in dirs and os.path.basename(root) in ("Train", "Validation", "Test"):
            os.symlink(root, f"/tmp/tdsc/{os.path.basename(root)}")
        for f in files:                                               # resume: an earlier version's output, added as an input
            if f.startswith("yu2024"):
                shutil.copy(os.path.join(root, f), f"{OUT}/repo/data/" if f == "yu2024.npz" else OUT)
    assert len(os.listdir("/tmp/tdsc")) == 3, "missing a split"


if __name__ == "__main__":
    import torch
    assert torch.cuda.device_count() == 2, f"expected T4 x 2, found {torch.cuda.device_count()} GPUs"
    setup()
    os.environ["TDSC_PATH"] = "/tmp/tdsc"
    sys.path.insert(0, f"{OUT}/repo")
    from abus_classification.papers.yu2024 import main
    main(out_dir=OUT, hours=10)                                       # ends before the 12 h session limit, so the output is kept
    shutil.copy(f"{OUT}/repo/data/yu2024.npz", OUT)
    shutil.rmtree(f"{OUT}/repo")
