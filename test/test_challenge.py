import numpy as np
import pandas as pd
import pytest

from abus_classification import challenge as c


def test_published_tables_are_reproduced():
    # Paper Table 7 (overall), Table 4 (segmentation), Table 5 (classification), Table 6 (detection)
    overall = c.overall_scores(c.PUBLISHED)
    table7 = {"T1 SZU": 2.6932, "T2 POSTECH": 1.8854, "T3 HU": 1.7075, "T4 KTH": 1.1995, "T5 EPITA": 0.7758,
              "T6 FX": 0.5039, "T7 PR": 0.4738}
    assert list(overall.index) == list(table7)
    assert np.allclose(overall[list(table7)], list(table7.values()), atol=2e-4)
    s = c.task_scores(c.PUBLISHED)
    for team, col, value in [("T8 NV", "Seg_Score", 0.9723), ("T13 IARI", "Seg_Score", 0.1687),
                             ("T4 KTH", "Cls_Score", 0.7827), ("T6 FX", "Cls_Score", 0.0),
                             ("T3 HU", "Det_Score", 0.8323), ("T16 UA", "Det_Score", 0.0)]:
        assert s.loc[team, col] == pytest.approx(value, abs=2e-4)


def test_classification_threshold_and_positive_class():
    m = c.classification([1, 1, 0, 0], [0.5, 0.4, 0.3, 0.6])       # 0.5 counts as malignant
    assert m["ACC"] == 0.5 and m["AUC"] == 0.5


def test_dice_hd_by_hand():
    gt, pred = np.zeros((20, 20, 20)), np.zeros((20, 20, 20))
    gt[2:4, 2:4, 2:4] = 1                                           # 8 voxels
    pred[2:4, 2:4, 3:5] = 1                                         # shifted by one, 4 shared
    dice, hd = c.dice_hd(gt, pred)
    assert dice == 0.5 and hd == 1
    assert c.dice_hd(gt, pred, spacing=(1, 1, 3))[1] == 3
    far = np.zeros_like(gt); far[2, 2, 18] = 1                      # farthest gt voxel (3, 3, 2): sqrt(1 + 1 + 16^2)
    assert c.dice_hd(gt, far)[1] == pytest.approx(np.sqrt(258))
    assert c.dice_hd(gt, np.zeros_like(gt)) == (0.0, float("inf"))


def test_inf_penalty():
    assert c.segmentation([(1, 10), (0, float("inf"))]) == {"DICE": 0.5, "HD": (10 + 10.5) / 2, "empty_cases": 1}
    assert c.segmentation([(1, 10), (0, float("inf"))], worst_hd=100)["HD"] == (10 + 105) / 2


def test_mask_box_matches_csv_layout():
    mask = np.zeros((30, 20, 10)); mask[5:16, 4:9, 2:5] = 1
    assert c.mask_box(mask) == (3.0, 6.0, 10.0, 2, 4, 10)


def test_iou_by_hand():
    assert c.iou_3d((0, 0, 0, 2, 2, 2), (0, 0, 0, 2, 2, 2)) == 1
    assert c.iou_3d((0, 0, 0, 2, 2, 2), (1, 0, 0, 2, 2, 2)) == pytest.approx(4 / 12)
    assert c.iou_3d((0, 0, 0, 2, 2, 2), (5, 0, 0, 2, 2, 2)) == 0


def test_froc_by_hand():
    box = (0, 0, 0, 10, 10, 10)
    gt = {1: [box], 2: [box]}
    assert c.froc(gt, {1: [(box, 0.9)], 2: [(box, 0.9)]})["FROC"] == pytest.approx(1)
    assert c.froc(gt, {})["FROC"] == pytest.approx(0, abs=1e-6)
    # one hit of two, no false positives: recall 0.5 at every level
    assert c.froc(gt, {1: [(box, 0.9)]})["FROC"] == pytest.approx(0.5)
    # IoU exactly at or below 0.3 is a miss; a duplicate hit is neither a second hit nor a false positive
    miss = (9.5, 0, 0, 10, 10, 10)
    assert c.iou_3d(box, miss) < 0.3 and c.froc({1: [box]}, {1: [(miss, 0.9)]})["FROC"] == pytest.approx(0, abs=1e-6)
    assert c.froc({1: [box]}, {1: [(box, 0.9), (box, 0.8)]})["FROC"] == pytest.approx(1)
    with pytest.raises(ValueError):
        c.froc(gt, {3: []})


def test_benchmark_row_on_the_published_board():
    best = c.benchmark({"m": {"DICE": 0.9, "HD": 1, "ACC": 0.99, "AUC": 0.99, "FROC": 0.99}}).loc["m"]
    assert best.Overall == 3 and best.Overall_Rank == "1/8" and best.Cls_Rank == "1/9"
    partial = c.benchmark({"m": {"ACC": 0.7, "AUC": 0.8}}).loc["m"]
    assert partial.Cls_Rank == "4/9" and partial.Seg_Rank == "" and np.isnan(partial.Overall)
    assert isinstance(partial.Cls_Score, float) and pd.isna(partial.Seg_Score)
