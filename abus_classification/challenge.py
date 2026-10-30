"""TDSC-ABUS 2023 challenge evaluation: the benchmark every reproduced paper reports next to its own protocol.

Follows the paper (Luo et al., arXiv:2501.15588, Sec. 2.3, 3.3) and the organisers' code
(github.com/PerceptionComputingLab/TDSC-ABUS2023, Final_Evaluation/):
    classification  ACC at p >= 0.5 and AUC, malignant = positive
    segmentation    mean Dice and mean symmetric Hausdorff distance per case, in voxels (the NRRD files store unit
                    spacing); an empty prediction has HD = inf, replaced by 1.05 x the worst valid HD
    detection       FROC: mean recall at 0.125-8 false positives per scan, a hit is IoU > 0.3
    scores          min-max normalisation over the valid teams; overall score over teams with all three tasks
Boxes are (coordX, coordY, coordZ, x_length, y_length, z_length); X, Y, Z are array axes 2, 1, 0 (see `mask_box`).
"""
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import accuracy_score, roc_auc_score

KEY_FP = (0.125, 0.25, 0.5, 1, 2, 4, 8)
IOU_HIT = 0.3
METRICS = ["DICE", "HD", "ACC", "AUC", "FROC"]

# Final test scores: Tables 4 (HD with the inf penalty), 5 and 6. T11, T14 and T17 have no reported scores.
PUBLISHED = pd.DataFrame.from_dict({
    "T1 SZU": (0.5861, 117.1939, 0.7571, 0.8892, 0.8468),
    "T2 POSTECH": (0.6147, 90.5339, 0.6429, 0.6558, 0.7303),
    "T3 HU": (0.5616, 162.9371, 0.7286, 0.7733, 0.7704),
    "T4 KTH": (0.5890, 159.0311, 0.7429, 0.7708, 0.6067),
    "T5 EPITA": (0.5377, 96.5050, 0.5429, 0.5775, 0.6383),
    "T6 FX": (0.5400, 121.1640, 0.5429, 0.5675, 0.6153),
    "T7 PR": (0.4981, 153.0743, 0.6000, 0.6425, 0.6441),
    "T8 NV": (0.6020, 82.8654, np.nan, np.nan, np.nan),
    "T9 UDG": (0.5853, 80.1817, np.nan, np.nan, 0.6459),
    "T10 DGIST": (0.5342, 105.0751, 0.7143, 0.7642, np.nan),
    "T12 ICL": (0.4665, 266.1207, np.nan, np.nan, np.nan),
    "T13 IARI": (0.3057, 203.4005, np.nan, np.nan, np.nan),
    "T15 MPU": (np.nan, np.nan, np.nan, np.nan, 0.5327),
    "T16 UA": (0.4412, 101.4036, np.nan, np.nan, 0.3913),
    "T18 XJTLU": (0.4890, 81.7367, np.nan, np.nan, np.nan),
}, orient="index", columns=METRICS)


def classification(y, p):
    """ACC at p >= 0.5 and AUC; y: 1 = malignant, p: predicted probability of malignant."""
    y, p = np.asarray(y), np.asarray(p)
    return {"ACC": accuracy_score(y, p >= 0.5), "AUC": roc_auc_score(y, p)}


def dice_hd(gt, pred, spacing=(1, 1, 1)):
    """Dice and symmetric Hausdorff distance of two masks of one case; HD is inf if either mask is empty."""
    gt, pred = np.asarray(gt) > 0, np.asarray(pred) > 0
    if gt.shape != pred.shape:
        raise ValueError(f"mask shapes differ: {gt.shape} vs {pred.shape}")
    dice = 2 * (gt & pred).sum() / max(gt.sum() + pred.sum(), 1)
    if not gt.any() or not pred.any():
        return float(dice), float("inf")
    points = np.argwhere(gt | pred)
    box = tuple(slice(a, b + 1) for a, b in zip(points.min(0), points.max(0)))      # every distance lies inside it
    gt, pred = gt[box], pred[box]
    hd = max(distance_transform_edt(~pred, sampling=spacing)[gt].max(),
             distance_transform_edt(~gt, sampling=spacing)[pred].max())
    return float(dice), float(hd)


def segmentation(cases, worst_hd=None):
    """Mean DICE and HD over (dice, hd) cases. An inf HD becomes 1.05 x worst_hd, by default the worst finite HD here.

    The challenge took the worst HD over all teams, which per-case scores are not published for, so a single method
    can only be penalised with its own worst case or a value passed in; this flatters a method with many empty cases.
    """
    dice, hd = np.asarray(cases, float).T
    finite = hd[np.isfinite(hd)]
    worst = worst_hd if worst_hd is not None else finite.max() if len(finite) else np.nan
    return {"DICE": dice.mean(), "HD": np.where(np.isfinite(hd), hd, 1.05 * worst).mean(),
            "empty_cases": int((~np.isfinite(hd)).sum())}


def mask_box(mask):
    """The ground-truth box of a mask, in the layout of `bbx_labels.csv` and the challenge's detection CSV."""
    points = np.argwhere(np.asarray(mask) > 0)
    lo, hi = points.min(0)[::-1], points.max(0)[::-1]                # array axes (0, 1, 2) are (Z, Y, X)
    return (*((lo + hi) / 2), *(hi - lo))


def iou_3d(a, b):
    """Intersection over union of two centre-and-length boxes."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    overlap = np.minimum(a[:3] + a[3:] / 2, b[:3] + b[3:] / 2) - np.maximum(a[:3] - a[3:] / 2, b[:3] - b[3:] / 2)
    if (overlap <= 0).any():
        return 0.0
    inter = overlap.prod()
    return float(inter / (a[3:].prod() + b[3:].prod() - inter))


def froc(gt, predictions):
    """Detection score. gt: {case: [box, ...]}; predictions: {case: [(box, probability), ...]} for any of those cases.

    Follows the organisers' code: confidence thresholds 0, 0.005, ..., 0.995; a prediction with IoU > 0.3 to its best
    ground truth is a hit, several hits on one ground truth count once and are not false positives.
    """
    if extra := set(predictions) - set(gt):
        raise ValueError(f"predictions for unknown cases: {sorted(extra)}")
    rows = []
    for case, boxes in gt.items():
        for box, prob in predictions.get(case, []):
            ious = [iou_3d(g, box) for g in boxes]
            rows.append((case, int(np.argmax(ious)), max(ious), prob))
    table = pd.DataFrame(rows, columns=["case", "gt", "iou", "prob"])
    eps, total_gt, fp, recall = 1e-8, sum(map(len, gt.values())), [], []
    for t in np.arange(0, 1, 0.005):
        kept = table[table.prob >= t]
        hits = kept[kept.iou > IOU_HIT]
        fp.append((len(kept) - len(hits) + eps) / (len(gt) + eps))
        recall.append((len(hits.drop_duplicates(["case", "gt"])) + eps) / (total_gt + eps))
    curve = pd.DataFrame({"fp": fp, "recall": recall}).sort_values("fp")
    key_recall = []
    for key in KEY_FP:
        less, more = curve[curve.fp <= key], curve[curve.fp >= key]
        if less.empty:
            key_recall.append(0.0)
        elif more.empty:
            key_recall.append(curve.recall.max())
        else:
            f0, f1, r0, r1 = less.fp.iloc[-1], more.fp.iloc[0], less.recall.iloc[-1], more.recall.iloc[0]
            key_recall.append(r0 + (r1 - r0) * (key - f0) / (f1 - f0 + eps))
    return {"FROC": float(np.mean(key_recall)), "key_recall": dict(zip(KEY_FP, key_recall)), "fp": fp, "recall": recall}


def _norm(x):
    return (x - x.min()) / (x.max() - x.min())


def task_scores(table):
    """Segmentation, classification and detection scores, each normalised over the teams with that task."""
    out = pd.DataFrame(index=table.index, columns=["Seg_Score", "Cls_Score", "Det_Score"], dtype=float)
    seg, cls, det = table[["DICE", "HD"]].dropna(), table[["ACC", "AUC"]].dropna(), table["FROC"].dropna()
    out["Seg_Score"] = (1 + _norm(seg.DICE) - _norm(seg.HD)) / 2
    out["Cls_Score"] = (_norm(cls.ACC) + _norm(cls.AUC)) / 2
    out["Det_Score"] = _norm(det)
    return out


def overall_scores(table):
    """Eq. 1, re-normalised over the teams with all three tasks; highest first."""
    t = table[METRICS].dropna()
    return ((1 + _norm(t.DICE) - _norm(t.HD)) / 2 + (_norm(t.ACC) + _norm(t.AUC)) / 2 + _norm(t.FROC)).sort_values(
        ascending=False).rename("Overall")


def benchmark(results):
    """Place our methods on the published leaderboards as if each had entered alone.

    results: {method: {metric: value}} with any of METRICS; a task with missing metrics is left out for that method.
    Returns one row per method with its metrics, task scores, ranks and the overall score where all tasks exist.
    """
    rows = []
    for name, values in results.items():
        table = pd.concat([PUBLISHED, pd.DataFrame({name: values}, index=METRICS).T.astype(float)])
        scores, row = task_scores(table), {m: values.get(m, np.nan) for m in METRICS}
        for task in scores:
            row[task] = scores.loc[name, task]
            valid = scores[task].dropna()
            row[f"{task[:3]}_Rank"] = f"{int(valid.rank(ascending=False, method='min')[name])}/{len(valid)}" if name in valid else ""
        overall = overall_scores(table)
        row["Overall"] = overall.get(name, np.nan)
        row["Overall_Rank"] = f"{list(overall.index).index(name) + 1}/{len(overall)}" if name in overall else ""
        rows.append(pd.Series(row, name=name))
    return pd.DataFrame(rows)
