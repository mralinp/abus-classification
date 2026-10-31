"""Keep a run's output on Google Drive through rclone, so that a cancelled or crashed Kaggle kernel loses little."""
import os
import subprocess
import threading


def configure_rclone(conf, path="/tmp/rclone.conf"):
    """Write rclone's config, which holds the Drive token, readable only by this user. Never print it."""
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        f.write(conf)
    os.environ["RCLONE_CONFIG"] = path


class DriveSync:
    """On entry: check Drive works (raises if not) and download earlier output. Inside: upload every `every` seconds. On exit: upload.

    Uploads are incremental and a failed one is only reported, so a Drive problem never stops training.
    """
    # ponytail: finished runs' checkpoints stay on Drive and are downloaded on resume, delete them if space or start-up time matters

    def __init__(self, folder, local, remote="gdrive", every=600):
        self.target, self.local, self.every = f"{remote}:{folder}", str(local), every
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def rclone(self, *args):
        return subprocess.run(["rclone", *args], capture_output=True, text=True)

    def push(self):
        done = self.rclone("copy", self.local, self.target, "--exclude", "repo/**", "--exclude", "*.tmp")
        print("drive: uploaded" if done.returncode == 0 else f"drive: upload failed: {done.stderr[-300:]}", flush=True)

    def loop(self):
        while not self.stop.wait(self.every):
            self.push()

    def __enter__(self):
        for args in (("mkdir", self.target), ("touch", f"{self.target}/.write-check"), ("lsf", f"{self.target}/.write-check"),
                     ("copy", self.target, self.local, "--exclude", "*.pt", "--exclude", ".write-check")):
            done = self.rclone(*args)
            if done.returncode != 0:
                raise RuntimeError(f"Drive is not usable, rclone {args[0]} failed: {done.stderr[-300:]}")
        print(f"drive: ready, {self.target}", flush=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.push()
