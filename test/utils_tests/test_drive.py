import os
import stat
import time

import pytest

from abus_classification.utils.drive import DriveSync, configure_rclone


@pytest.fixture
def rclone(tmp_path, monkeypatch):
    """A stand-in rclone that records its arguments and fails when told to."""
    calls, script = tmp_path / "calls", tmp_path / "rclone"
    script.write_text(f'#!/bin/sh\necho "$@" >> {calls}\n[ -f {tmp_path}/fail-$1 ] && {{ echo boom >&2; exit 1; }}\nexit 0\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    return calls


def test_checks_downloads_then_uploads_on_a_timer_and_at_exit(rclone, tmp_path):
    with DriveSync("runs/a", tmp_path, every=0.05):
        time.sleep(0.3)
    lines = rclone.read_text().splitlines()
    assert [l.split()[0] for l in lines[:3]] == ["mkdir", "touch", "lsf"]
    assert lines[3].startswith(f"copy gdrive:runs/a {tmp_path}") and "*.pt" in lines[3]
    uploads = [l for l in lines if l.startswith(f"copy {tmp_path} gdrive:runs/a")]
    assert len(uploads) >= 2 and "repo/**" in uploads[0] and "*.tmp" in uploads[0]


def test_an_unusable_drive_stops_the_run_at_the_start(rclone, tmp_path):
    (tmp_path / "fail-touch").write_text("")
    with pytest.raises(RuntimeError, match="touch failed: boom"):
        DriveSync("runs/a", tmp_path).__enter__()


def test_a_failed_upload_does_not_stop_training(rclone, tmp_path, capsys):
    sync = DriveSync("runs/a", tmp_path)
    (tmp_path / "fail-copy").write_text("")
    sync.push()
    assert "upload failed: boom" in capsys.readouterr().out


def test_config_is_private(tmp_path, monkeypatch):
    path = tmp_path / "rclone.conf"
    monkeypatch.delenv("RCLONE_CONFIG", raising=False)
    configure_rclone("[gdrive]\n", str(path))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and os.environ["RCLONE_CONFIG"] == str(path)
