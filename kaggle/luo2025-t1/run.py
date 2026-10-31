import os, shutil, subprocess, sys
OUT = "/kaggle/working"
sh = lambda c: subprocess.run(c, shell=True, check=True)


def setup():
    sh(f"git clone -q -b feat/luo2025-t1 https://github.com/mralinp/abus-classification.git {OUT}/repo")
    sh("pip install -q pynrrd trimesh tdsc-abus2023-pytorch")
    sh("curl -fsSL -o /tmp/rclone.zip https://downloads.rclone.org/v1.68.2/rclone-v1.68.2-linux-amd64.zip"
       " && unzip -q -j -o /tmp/rclone.zip '*/rclone' -d /usr/local/bin")
    os.makedirs("/tmp/tdsc", exist_ok=True)
    for root, dirs, _ in os.walk("/kaggle/input"):
        if "DATA" in dirs and os.path.basename(root) in ("Train", "Validation", "Test"):
            os.symlink(root, f"/tmp/tdsc/{os.path.basename(root)}")
    assert len(os.listdir("/tmp/tdsc")) == 3, "missing a split"


if __name__ == "__main__":
    import torch
    assert torch.cuda.device_count() == 2, f"expected T4 x 2, found {torch.cuda.device_count()} GPUs"
    setup()
    os.environ["TDSC_PATH"] = "/tmp/tdsc"
    sys.path.insert(0, f"{OUT}/repo")
    from kaggle_secrets import UserSecretsClient
    from abus_classification.papers.luo2025_t1 import main
    from abus_classification.utils.drive import DriveSync, configure_rclone
    configure_rclone(UserSecretsClient().get_secret("RCLONE_CONF"))
    with DriveSync("abus-checkpoints/luo2025-t1", OUT):               # fails here, in minutes, if Drive does not work
        main(out_dir=OUT, hours=10)                                   # ends before the 12 h session limit, so the output is kept
    shutil.rmtree(f"{OUT}/repo")
