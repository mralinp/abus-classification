import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("kaggle_push", Path(__file__).parents[1] / "kaggle" / "push.py")
push = importlib.util.module_from_spec(spec)
spec.loader.exec_module(push)


def test_owners_come_from_the_local_map_with_the_kernel_account_as_default():
    meta = {"id": "me/run", "dataset_sources": ["shared", "mine"], "kernel_sources": ["cache"], "title": "run"}
    out = push.resolve(meta, {"shared": "other"})
    assert out["dataset_sources"] == ["other/shared", "me/mine"] and out["kernel_sources"] == ["me/cache"]
    assert out["id"] == "me/run" and out["title"] == "run"
