"""Can the signs of malignancy be read from a lesion's 3D surface? Descriptors designed from the clinical signs, on TDSC-ABUS.

Features are computed for all 200 lesions from geometry and the image alone. The analysis reads Train+Validation only.
Test was used once by `lb_evaluation`, so `--evaluate` is a second look at that split: `--freeze` fits and names every model and
check on Train+Validation first, and `--evaluate` applies the stored coefficients once.

    python -m abus_classification.lb_malignancy
    python -m abus_classification.lb_malignancy --predicted   # Zhou 2021 masks on Test patches, geometry and image only
    python -m abus_classification.lb_malignancy --control     # sham lesions, reach out to 30 mm, dark lines
    python -m abus_classification.lb_malignancy --analyse
    python -m abus_classification.lb_malignancy --samples     # meshes and planes for the figures
    python -m abus_classification.lb_malignancy --freeze      # Train+Validation: the models and checks for Test
    python -m abus_classification.lb_malignancy --evaluate    # once, on Test
"""
import json
import os
import sys
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"   # one process per lesion

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score
from tdsc_abus2023_pytorch import TDSC, DataSplits

from abus_classification import challenge
from abus_classification.features.shape_descriptor.malignancy_signs import (
    LOBE_PROMINENCE, SHELL_MM, SHELLS_MM, line_field, line_points, line_summary, line_surface_field, line_surface_summary, mesh_spectrum, radial_lines,
    surface_height, surface_peaks, surface_signs)
from abus_classification.features.tan.spiculation import line_orientation
from abus_classification.lb_classification import build_table, cv_auc, delta, forest, groups_of, linear, nested_selection, oof, oof_mean, splits
from abus_classification.lb_descriptors import DRAWS, NOISE_MM, RENDERS, stack
from abus_classification.lb_invariance import TDSC_PATH, largest_component, tight
from abus_classification.lb_roundness import CELLS
from abus_classification.lb_stability import boundary_noise, on_grid, predicted_masks, surface
from abus_classification.protocol import auc_ci
from abus_classification.utils.spacing import TDSC_DEPTH_AXIS, TDSC_SPACING, prepare_lesion

OUT = Path(__file__).resolve().parents[1] / "results" / "lb-malignancy"
K, LARGE_LOBE = 200, 0.15     # at 0.15 of the radius the median lesion has about four lobes, the benign limit of Stavros et al. (1995)
HYBRID = ["Size: log volume", "Radial lines: mean", "Orientation: tilt"]     # what the inside-fold procedure picked, less the terms that added nothing
LB_TERM = "Irregularity: log budget, degrees 8-13"
WIDER = "Wider shell: lines over 4-30 mm"
PLAN, PRIMARY = OUT / "test_plan.json", "Hybrid: size, radial lines, tilt"
SHAM_MARGIN_MM, CONTROL_SHELLS_MM = 70.0, (2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0)


def fields(grid_mask, grid, side, crop_mask, crop_spacing, theta, strength):
    """Mesh, spectrum, pieces, lines, line field on the surface and height above the body, for one version of a lesion."""
    mesh, pieces = surface(grid_mask, grid, side)
    spectrum = mesh_spectrum(mesh, K)
    lines = radial_lines(theta, strength, crop_mask, crop_spacing, TDSC_DEPTH_AXIS)
    # The two masks live on different grids, so the line points are moved onto the mesh by the lesion's centroid.
    points = line_points(lines, crop_spacing, TDSC_DEPTH_AXIS) - np.argwhere(crop_mask).mean(0) * crop_spacing + mesh.center_mass
    height = surface_height(mesh, spectrum["phi"], spectrum["c"]) / np.sqrt(mesh.area / (4 * np.pi))
    return mesh, spectrum, pieces, lines, line_surface_field(points, lines, mesh, spectrum), height


def measure(grid_mask, grid, side, crop_mask, crop_spacing, theta, strength):
    """Every descriptor of one version of a lesion: a mask on its size-matched grid, and on the image's crop."""
    mesh, spectrum, pieces, lines, field, height = fields(grid_mask, grid, side, crop_mask, crop_spacing, theta, strength)
    return dict(surface_signs(mesh, spectrum, TDSC_DEPTH_AXIS), **line_summary(lines, np.argwhere(crop_mask)[:, TDSC_DEPTH_AXIS].mean()),
                **line_surface_summary(field, height, spectrum, mesh), volume=abs(mesh.volume), area=mesh.area,
                vertices=len(mesh.vertices), pieces=pieces)


def one_lesion(split, index, seed):
    volume, mask, label, _ = TDSC(path=TDSC_PATH, split=split, download=False)[index]
    mask = np.asarray(mask) > 0
    crop, crop_mask, crop_spacing = prepare_lesion(volume, mask)
    del volume
    theta, strength = line_field(crop, crop_spacing, TDSC_DEPTH_AXIS)
    mask = tight(largest_component(tight(mask)))
    side = (mask.sum() * np.prod(TDSC_SPACING)) ** (1 / 3)
    clean, grid = on_grid(mask, TDSC_SPACING, side, CELLS)
    rows = [measure(clean, grid, side, crop_mask, crop_spacing, theta, strength)]
    for i, amplitude in enumerate(NOISE_MM):
        for draw in range(DRAWS):
            noisy = seed + 100 * i + draw
            rows.append(measure(boundary_noise(clean, grid, amplitude, noisy), grid, side,
                                largest_component(boundary_noise(crop_mask, crop_spacing, amplitude, noisy)), crop_spacing, theta, strength))
    return dict(stack(rows), y=int(label == 0), test=split == DataSplits.TEST)


def main():
    jobs = []
    for split in (DataSplits.TRAIN, DataSplits.VALIDATION, DataSplits.TEST):
        jobs += [(split, i, 1000 * len(jobs) + 1000 * i) for i in range(len(TDSC(path=TDSC_PATH, split=split, download=False)))]
    cases = Parallel(n_jobs=-1, verbose=5)(delayed(one_lesion)(*j) for j in jobs)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "features.npz", renders=np.array(RENDERS), **{k: np.asarray([c[k] for c in cases]) for k in cases[0]})


def one_prediction(volume, truth, predictions, spacing=np.array((0.476, 0.476, 0.476))):
    theta, strength = line_field(volume, spacing, TDSC_DEPTH_AXIS)
    truth = largest_component(truth)
    side = (truth.sum() * np.prod(spacing)) ** (1 / 3)
    rows, template = [], None
    for mask in [truth] + list(predictions):
        try:
            mask = largest_component(mask)
            grid_mask, grid = on_grid(tight(mask), spacing, side, CELLS)
            row = measure(grid_mask, grid, side, mask, spacing, theta, strength)
            template = template or {k: np.full(np.shape(v), np.nan) for k, v in row.items()}
        except (ValueError, IndexError):    # nothing found, or too small to survive the smoothing: counted as a miss
            row = None
        rows.append(row)
    return stack([template if r is None else r for r in rows])


def predicted():
    from abus_classification.data.lesion_cache import load_raw
    masks, truth, _ = predicted_masks()
    data, _, test = load_raw()
    volumes = np.asarray(data["vol"][test], np.float32)
    rows = Parallel(n_jobs=-1, verbose=5)(delayed(one_prediction)(volumes[i], truth[i], list(masks[:, i])) for i in range(len(truth)))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "predicted.npz", **{k: np.asarray([r[k] for r in rows]) for k in rows[0]})


def sham_masks(mask, spacing, gap_mm=20.0, shell_mm=SHELL_MM[1]):
    """The lesion's mask moved in the coronal plane to tissue away from it, in each of four directions that stay in the crop."""
    m = np.moveaxis(mask, TDSC_DEPTH_AXIS, 0)
    pixel = np.delete(np.asarray(spacing), TDSC_DEPTH_AXIS)
    pts = np.argwhere(m)[:, 1:]
    low, high = pts.min(0), pts.max(0)
    shams = []
    for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):
        shift = sign * int(np.ceil(((high - low)[axis] * pixel[axis] + gap_mm + 2 * shell_mm) / pixel[axis]))
        margin = int(np.ceil(shell_mm / pixel[axis]))
        if low[axis] + shift - margin >= 0 and high[axis] + shift + margin < m.shape[axis + 1]:
            shams.append(np.moveaxis(np.roll(m, shift, axis=axis + 1), 0, TDSC_DEPTH_AXIS))
    return shams


def one_control(split, index):
    """Radial lines at the lesion out to 30 mm, on sham lesions in the same scan, and for dark instead of bright lines."""
    volume, mask, label, _ = TDSC(path=TDSC_PATH, split=split, download=False)[index]
    crop, crop_mask, spacing = prepare_lesion(volume, np.asarray(mask) > 0, lateral_margin_mm=SHAM_MARGIN_MM)
    del volume
    theta, strength = line_field(crop, spacing, TDSC_DEPTH_AXIS)
    planes = np.argwhere(crop_mask)[:, TDSC_DEPTH_AXIS].mean()

    def summary(m, lines_theta=theta, lines_strength=strength):
        lines = radial_lines(lines_theta, lines_strength, m, spacing, TDSC_DEPTH_AXIS, shell_mm=(CONTROL_SHELLS_MM[0], CONTROL_SHELLS_MM[-1]))
        return line_summary(lines[lines[:, 3] < SHELL_MM[1]], planes)["radial"], line_summary(lines, planes, CONTROL_SHELLS_MM)["radial_by_shell"]

    radial, by_shell = summary(crop_mask)
    shams = [summary(m) for m in sham_masks(crop_mask, spacing)]
    pixel = float(np.mean(np.delete(spacing, TDSC_DEPTH_AXIS)))
    dark_theta, dark_strength = line_orientation(np.moveaxis(crop, TDSC_DEPTH_AXIS, 0), [0.9 / pixel], polarity="dark")
    nan = np.full(len(CONTROL_SHELLS_MM) - 1, np.nan)
    return dict(radial=radial, by_shell=by_shell, sham_radial=np.mean([a for a, _ in shams]) if shams else np.nan, sham_by_shell=np.mean([b for _, b in shams], axis=0) if shams else nan,
                shams=len(shams), dark_radial=summary(crop_mask, dark_theta, dark_strength)[0], y=int(label == 0), test=split == DataSplits.TEST)


def control():
    jobs = []
    for split in (DataSplits.TRAIN, DataSplits.VALIDATION, DataSplits.TEST):
        jobs += [(split, i) for i in range(len(TDSC(path=TDSC_PATH, split=split, download=False)))]
    cases = Parallel(n_jobs=8, verbose=5)(delayed(one_control)(*j) for j in jobs)     # the larger crops need more memory per process
    np.savez_compressed(OUT / "control.npz", shells=np.array(CONTROL_SHELLS_MM), **{k: np.asarray([c[k] for c in cases]) for k in cases[0]})


def samples():
    """Meshes, fields and the widest coronal plane for the figures: Train lesions at volume percentiles 25, 50 and 75 of each class."""
    d = np.load(OUT / "features.npz")
    data = TDSC(path=TDSC_PATH, split=DataSplits.TRAIN, download=False)
    out = {}
    for label, name in ((1, "malignant"), (0, "benign")):
        members = np.flatnonzero(d["y"][:len(data)] == label)
        volumes = d["volume"][members, 0]
        for q in (25, 50, 75):
            index = int(members[np.abs(volumes - np.percentile(volumes, q)).argmin()])
            volume, mask, _, _ = data[index]
            mask = np.asarray(mask) > 0
            crop, crop_mask, crop_spacing = prepare_lesion(volume, mask)
            theta, strength = line_field(crop, crop_spacing, TDSC_DEPTH_AXIS)
            mask = tight(largest_component(tight(mask)))
            side = (mask.sum() * np.prod(TDSC_SPACING)) ** (1 / 3)
            mesh, spectrum, _, _, field, height = fields(*on_grid(mask, TDSC_SPACING, side, CELLS), side, crop_mask, crop_spacing, theta, strength)
            peaks, prominence = surface_peaks(height, mesh)
            m = np.moveaxis(crop_mask, TDSC_DEPTH_AXIS, 0)
            plane = int(m.sum(axis=(1, 2)).argmax())
            rows, cols = np.nonzero(m[plane])
            pad = int(14 / np.delete(crop_spacing, TDSC_DEPTH_AXIS)[0])
            box = (slice(max(rows.min() - pad, 0), rows.max() + pad), slice(max(cols.min() - pad, 0), cols.max() + pad))
            key = f"{name}{q}"
            out.update({f"{key}_vertices": mesh.vertices.astype(np.float32), f"{key}_faces": mesh.faces.astype(np.int32), f"{key}_height": height.astype(np.float32),
                        f"{key}_field": field.astype(np.float32), f"{key}_peaks": peaks[prominence >= LOBE_PROMINENCE].astype(np.int32),
                        f"{key}_image": np.moveaxis(crop, TDSC_DEPTH_AXIS, 0)[plane][box].astype(np.float32), f"{key}_mask": m[plane][box], f"{key}_theta": theta[plane][box].astype(np.float32),
                        f"{key}_strength": strength[plane][box].astype(np.float32), f"{key}_pixel": np.delete(crop_spacing, TDSC_DEPTH_AXIS), f"{key}_index": index})
    np.savez_compressed(OUT / "samples.npz", **out)


def descriptor_columns(d):
    """The designed descriptors of a features file, as {'<family>: <name>': values}, for its first two axes (lesions, renders)."""
    budget = d["budget"]
    return {
        "Irregularity: log 1/roundness": -np.log(d["roundness"]),
        "Irregularity: log bending": np.log(d["bending"]),
        "Irregularity: log budget, degrees 2-3": np.log(budget[..., 1:3].sum(-1)),
        "Irregularity: log budget, degrees 4-7": np.log(budget[..., 3:7].sum(-1)),
        "Irregularity: log budget, degrees 8-13": np.log(budget[..., 7:13].sum(-1)),
        "Lobes: log lobes": np.log1p(d["lobes"]),
        "Lobes: log large lobes": np.where(np.isnan(d["prominence"]), np.nan, np.log1p((d["top_prominence"] >= LARGE_LOBE).sum(-1))),
        "Lobes: log prominence": np.log(d["prominence"]),
        "Lobes: log roughness": np.log(d["roughness"]),
        "Lobes: log angularity": np.log(d["angularity"]),
        "Orientation: logit depth share": np.log(d["depth_share"] / (1 - d["depth_share"])),
        "Orientation: tilt": d["tilt"],
        "Radial lines: mean": d["radial"],
        "Radial lines: skin side": d["radial_upper"],
        "Radial lines: strongest tenth": d["radial_p90"],
        "Radial lines: share outward": d["radial_share"],
    }


def table():
    """Earlier features and the new descriptors, clean surfaces, all 200 lesions in dataset order."""
    d = np.load(OUT / "features.npz")
    base = build_table()
    assert (base.y.values == d["y"]).all() and ((base.split == "Test").values == d["test"]).all()
    new = pd.DataFrame({k: v[:, 0] for k, v in descriptor_columns(d).items()})
    return pd.concat([base, new], axis=1)


def after(x, volume):
    v = np.c_[np.ones(len(volume)), volume]
    return x - v @ np.linalg.lstsq(v, x, rcond=None)[0]


def auc_either(y, x):
    auc = roc_auc_score(y, x)
    return max(auc, 1 - auc), "+" if auc >= 0.5 else "-"


def icc(x):
    """One-way intraclass correlation over lesions (rows) and renders (columns)."""
    n, k = x.shape
    between = k * ((x.mean(1) - x.mean()) ** 2).sum() / (n - 1)
    within = ((x - x.mean(1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return float((between - within) / (between + (k - 1) * within))


def cv_acc(X, y, reps=20):
    return float(np.mean([accuracy_score(y, oof(X, y, s) > 0) for s in range(reps)]))


def shuffled_gain(base, new, y, seed, reps=5):
    """Gain from the same number of columns with no link to this lesion: the new columns, shuffled over lesions."""
    order = np.random.default_rng(seed).permutation(len(y))
    return roc_auc_score(y, oof_mean(np.c_[base, new[order]], y, reps)) - roc_auc_score(y, oof_mean(base, y, reps))


def family_picks(rows, groups, families, part):
    """The feature of each family with the highest AUC after volume, on these lesions only."""
    y, volume = rows.y.values[part], rows["Size: log volume"].values[part]
    return [max(groups[g], key=lambda c: auc_either(y, after(rows[c].values[part], volume))[0]) for g in families]


def small_model(rows, groups, families, seed):
    """One five-fold split of log volume plus one feature per family, chosen inside each training part."""
    y, p, picks = rows.y.values, np.zeros(len(rows)), []
    for tr, te in splits(y, seed):
        columns = ["Size: log volume"] + family_picks(rows, groups, families, tr)
        p[te] = linear().fit(rows[columns].values[tr], y[tr]).decision_function(rows[columns].values[te])
        picks += columns[1:]
    return p, picks


def analyse(reps=20, shuffles=200):
    all_rows = table()
    keep = (all_rows.split != "Test").values
    rows = all_rows[keep].reset_index(drop=True)
    rows.to_csv(OUT / "features.csv", index=False)
    y = rows.y.values
    d = np.load(OUT / "features.npz")
    columns = descriptor_columns(d)
    groups = groups_of(rows.columns)
    families = [g for g in groups if g in ("Irregularity", "Lobes", "Orientation", "Radial lines")]
    volume = rows["Size: log volume"].values
    size, spic = rows[groups["Size"]].values, rows[groups["Spiculation"]].values
    pool = Parallel(n_jobs=-1)

    single = {}
    for name in sum((groups[g] for g in families), []) + ["Classical shape: VHWR", "Classical shape: DF", "Elongation and sphericity: log elongation",
                                                          "Spiculation: SP90", "Spiculation: SPcyl_upper_mean", "Tips: log tips, tau 0.021", "Hersch ratio: lambda_1 * area / 8 pi"]:
        x = rows[name].values
        raw, sign = auc_either(y, x)
        row = dict(auc=raw, sign=sign, auc_after_volume=auc_either(y, after(x, volume))[0], rho_volume=spearmanr(x, volume)[0], rho_sp90=spearmanr(after(x, volume), after(rows["Spiculation: SP90"].values, volume))[0],
                   rho_vhwr=spearmanr(after(x, volume), after(rows["Classical shape: VHWR"].values, volume))[0])
        if name in columns:
            renders = columns[name][keep]
            row.update({f"icc_{a}": icc(renders[:, [0] + [1 + i * DRAWS + j for j in range(DRAWS)]]) for i, a in enumerate(NOISE_MM)})
        single[name] = row
    pd.DataFrame(single).T.to_csv(OUT / "single.csv")

    base_scores = oof_mean(np.c_[size, spic], y, reps)
    size_scores = oof_mean(size, y, reps)
    effects = {}
    for g in families:
        X = rows[groups[g]].values
        gain = delta(base_scores, oof_mean(np.c_[size, spic, X], y, reps), y)
        null = np.array(pool(delayed(shuffled_gain)(np.c_[size, spic], X, y, s) for s in range(shuffles)))
        effects[g] = dict(alone=cv_auc(X, y, reps), with_size=delta(size_scores, oof_mean(np.c_[size, X], y, reps), y), after_size=cv_auc(after(X, volume), y, reps),
                          add_to_base=gain, shuffled_gain_p95=float(np.percentile(null, 95)), shuffled_p=float((1 + (null >= gain["gain"]).sum()) / (1 + shuffles)))
    (OUT / "effects.json").write_text(json.dumps(effects, indent=1))

    designed = np.c_[tuple(rows[groups[g]].values for g in families)]
    surface_only = np.c_[tuple(rows[groups[g]].values for g in families if g != "Radial lines")]
    vhwr = rows[["Classical shape: VHWR"]].values
    models = {"Size": size, "Size + spiculation": np.c_[size, spic], "Size + VHWR": np.c_[size, vhwr], "Size + spiculation + VHWR": np.c_[size, spic, vhwr], "Size + surface signs (mask and beam)": np.c_[size, surface_only],
              "Size + radial lines": np.c_[size, rows[groups["Radial lines"]].values], "Size + spiculation + radial lines": np.c_[size, spic, rows[groups["Radial lines"]].values],
              "Size + orientation + radial lines": np.c_[size, rows[groups["Orientation"]].values, rows[groups["Radial lines"]].values],
              "Size + spiculation + orientation + radial lines": np.c_[size, spic, rows[groups["Orientation"]].values, rows[groups["Radial lines"]].values],
              "Size + all designed": np.c_[size, designed], "Size + spiculation + all designed": np.c_[size, spic, designed],
              "Surface signs alone (mask and beam)": surface_only, "Radial lines alone": rows[groups["Radial lines"]].values, "Spiculation alone": spic, "All designed, no size": designed,
              "Hybrid: size, radial lines, tilt": rows[HYBRID].values, "Hybrid + fine budget (LB)": rows[HYBRID + [LB_TERM]].values}
    pd.DataFrame({n: dict(n=X.shape[1], auc=cv_auc(X, y, reps), acc=cv_acc(X, y, reps), auc_forest=cv_auc(X, y, 5, forest)) for n, X in models.items()}).T.to_csv(OUT / "models.csv")
    pd.DataFrame({n: oof_mean(X, y, reps) for n, X in models.items()}).assign(y=y, case=rows.case.values, split=rows.split.values).to_csv(OUT / "scores.csv", index=False)

    comparisons = {"Size + radial lines vs size + spiculation": ("Size + spiculation", "Size + radial lines"), "Size + orientation + radial lines vs size + spiculation": ("Size + spiculation", "Size + orientation + radial lines"),
                   "Size + orientation + radial lines vs size": ("Size", "Size + orientation + radial lines"), "Size + surface signs vs size": ("Size", "Size + surface signs (mask and beam)"),
                   "Hybrid vs size + spiculation": ("Size + spiculation", "Hybrid: size, radial lines, tilt"), "Hybrid vs size": ("Size", "Hybrid: size, radial lines, tilt"),
                   "Fine budget added to the hybrid": ("Hybrid: size, radial lines, tilt", "Hybrid + fine budget (LB)")}
    scores = {n: oof_mean(X, y, reps) for n, X in models.items()}
    runs = pool(delayed(small_model)(rows, groups, families, seed) for seed in range(reps))
    small = np.mean([p for p, _ in runs], 0)
    counts = pd.Series(sum((c for _, c in runs), [])).value_counts() / (reps * 5)
    final = linear().fit(rows[HYBRID].values, y)
    null = np.array(pool(delayed(shuffled_gain)(rows[HYBRID].values, rows[[LB_TERM]].values, y, s) for s in range(shuffles)))
    fine_gain = roc_auc_score(y, scores["Hybrid + fine budget (LB)"]) - roc_auc_score(y, scores["Hybrid: size, radial lines, tilt"])
    frozen = dict(features=HYBRID, coefficients=final[-1].coef_[0].tolist(), intercept=float(final[-1].intercept_[0]), mean=final[0].mean_.tolist(), scale=final[0].scale_.tolist(),
                  picks_of_all_lesions=["Size: log volume"] + family_picks(rows, groups, families, np.arange(len(y))), procedure_auc=float(np.mean([roc_auc_score(y, p) for p, _ in runs])), procedure_acc=float(np.mean([accuracy_score(y, p > 0) for p, _ in runs])),
                  picked=counts.to_dict(), procedure_against_size_spiculation=delta(scores["Size + spiculation"], small, y), fine_budget_shuffled_p=float((1 + (null >= fine_gain).sum()) / (1 + shuffles)))
    (OUT / "frozen.json").write_text(json.dumps(frozen, indent=1))
    rows[["case", "split", "y"]].assign(score=small, **{n: scores[n] for n in ("Size", "Size + spiculation")}).to_csv(OUT / "small_scores.csv", index=False)
    (OUT / "comparisons.json").write_text(json.dumps({n: delta(scores[a], scores[b], y) for n, (a, b) in comparisons.items()}, indent=1))

    if (OUT / "control.npz").exists():
        c = np.load(OUT / "control.npz")
        ok = keep & np.isfinite(c["sham_radial"])
        at_lesion_vs_sham = delta(c["sham_radial"][ok], c["radial"][ok], c["y"][ok])
        sham_null = np.array([roc_auc_score(c["y"][ok][i], c["sham_radial"][ok][i]) for i in (lambda r: [r.integers(0, ok.sum(), ok.sum()) for _ in range(2000)])(np.random.default_rng(0))])
        control = dict(lesions=int(keep.sum()), with_sham=int(ok.sum()), median_shams=float(np.median(c["shams"][keep])), auc_lesion=auc_either(c["y"][keep], c["radial"][keep]), auc_lesion_same_lesions=auc_either(c["y"][ok], c["radial"][ok]), auc_sham=auc_either(c["y"][ok], c["sham_radial"][ok]),
                       sham_auc_low=float(np.percentile(sham_null, 2.5)), sham_auc_high=float(np.percentile(sham_null, 97.5)), lesion_minus_sham=at_lesion_vs_sham,
                       auc_dark=auc_either(c["y"][keep], c["dark_radial"][keep]), rho_stored=float(spearmanr(c["radial"][keep], d["radial"][keep, 0])[0]), rho_bright_dark=float(spearmanr(c["radial"][keep], c["dark_radial"][keep])[0]),
                       rho_lesion_sham=float(spearmanr(c["radial"][ok], c["sham_radial"][ok])[0]))
        (OUT / "control.json").write_text(json.dumps(control, indent=1))
        shells = [f"{a:g}-{b:g} mm" for a, b in zip(c["shells"][:-1], c["shells"][1:])]
        y_c = c["y"]
        pd.DataFrame({(place, "malignant" if label else "benign", s): dict(mean=np.nanmean(values[keep & (y_c == label), i]), se=np.nanstd(values[keep & (y_c == label), i]) / np.sqrt(np.isfinite(values[keep & (y_c == label), i]).sum()))
                      for place, values in (("lesion", c["by_shell"]), ("sham", c["sham_by_shell"])) for label in (1, 0) for i, s in enumerate(shells)}).T.to_csv(OUT / "control_shells.csv")
        # Exploratory: other shells, tried after the distance profile was seen. Plain means of the shells' excess.
        tilt = rows[HYBRID[:1] + HYBRID[2:]].values
        wider = {"2-10 mm, as designed": rows["Radial lines: mean"].values}
        wider.update({s: c["by_shell"][keep, i] for i, s in enumerate(shells)})
        wider.update({f"{a}-{b} mm": c["by_shell"][keep][:, shells.index(f"{a}-{a + 2} mm"):[s.split("-")[1] for s in shells].index(f"{b} mm") + 1].mean(1) for a, b in ((4, 10), (4, 20), (4, 30))})
        pd.DataFrame({n: dict(auc=auc_either(y, x)[0], auc_after_volume=auc_either(y, after(x, volume))[0], rho_volume=spearmanr(x, volume)[0], hybrid_auc=cv_auc(np.c_[tilt, x], y, reps), hybrid_acc=cv_acc(np.c_[tilt, x], y, reps)) for n, x in wider.items()}).T.to_csv(OUT / "reach.csv")

    if (OUT / "predicted.npz").exists():
        p = np.load(OUT / "predicted.npz")
        truth_vs_predicted = {}
        for name, values in descriptor_columns(p).items():
            seeds = []
            for s in range(1, values.shape[1]):
                ok = np.isfinite(values[:, 0]) & np.isfinite(values[:, s])
                seeds.append(spearmanr(values[ok, 0], values[ok, s])[0])
            truth_vs_predicted[name] = dict(rank=float(np.mean(seeds)), misses=int((~np.isfinite(values[:, 1:])).sum()))
        pd.DataFrame(truth_vs_predicted).T.to_csv(OUT / "predicted.csv")

    blocks, start = {}, 0
    names = ["Size", "Spiculation"] + families
    X = np.c_[tuple(rows[groups[g]].values for g in names)]
    for g in names:
        blocks[g] = list(range(start, start + len(groups[g])))
        start += len(groups[g])
    done = pool(delayed(nested_selection)(blocks, X, y, seed) for seed in range(reps))
    counts = {}
    for _, choices in done:
        for chosen in choices:
            for c in chosen:
                counts[c] = counts.get(c, 0) + 1 / (reps * 5)
    (OUT / "selection.json").write_text(json.dumps(dict(auc=float(np.mean([roc_auc_score(y, p) for p, _ in done])), picked=dict(sorted(counts.items(), key=lambda kv: -kv[1]))), indent=1))

    by_shell = d["radial_by_shell"][keep, 0]
    labels = [f"{a:g}-{b:g} mm" for a, b in zip(SHELLS_MM[:-1], SHELLS_MM[1:])]
    pd.DataFrame({("malignant" if c else "benign", s): dict(mean=by_shell[y == c, i].mean(), se=by_shell[y == c, i].std() / np.sqrt((y == c).sum()), positive=(by_shell[y == c, i] > 0).mean())
                  for c in (1, 0) for i, s in enumerate(labels)}).T.to_csv(OUT / "shells.csv")
    coupling = d["coupling"][keep, 0]
    overlap = {s: {c: spearmanr(after(rows[s].values, volume), after(rows[c].values, volume))[0] for g in families for c in groups[g]} for s in groups["Spiculation"] + groups["Intensity"] + groups["Classical shape"]}
    pd.DataFrame(overlap).to_csv(OUT / "overlap.csv")
    (OUT / "coupling.json").write_text(json.dumps(dict(malignant=float(np.median(coupling[y == 1])), benign=float(np.median(coupling[y == 0])), auc=auc_either(y, coupling), covered=float(np.median(d["covered"][keep, 0]))), indent=1))


def fit_model(rows, columns):
    """Logistic regression on standardised columns, stored as plain numbers."""
    model = linear().fit(rows[columns].values.astype(float), rows.y.values)
    return dict(columns=columns, mean=model[0].mean_.tolist(), scale=model[0].scale_.tolist(), coefficients=model[-1].coef_[0].tolist(), intercept=float(model[-1].intercept_[0]))


def decision(model, rows):
    """The stored model's log-odds of malignant; p >= 0.5 is decision >= 0."""
    z = (rows[model["columns"]].values.astype(float) - np.array(model["mean"])) / np.array(model["scale"])
    return model["intercept"] + z @ np.array(model["coefficients"])


def plan_table():
    """All 200 lesions, with the wider shell as a column."""
    rows, c = table(), np.load(OUT / "control.npz")
    assert (rows.y.values == c["y"]).all()
    rows[WIDER] = c["by_shell"][:, 1:].mean(1)
    return rows


def freeze():
    """Every model and check for Test, fitted and named on Train+Validation. Nothing here reads a Test row."""
    assert not PLAN.exists() or "--force" in sys.argv, "already frozen: changing it after a Test run would invalidate that run"
    rows = plan_table()
    rows = rows[rows.split != "Test"].reset_index(drop=True)
    assert len(rows) == 130
    g = groups_of(rows.columns)
    surface_signs_columns = g["Irregularity"] + g["Lobes"] + g["Orientation"]
    sets = {PRIMARY: HYBRID, "Size": g["Size"], "Size + spiculation": g["Size"] + g["Spiculation"], "Hybrid + fine budget (LB)": HYBRID + [LB_TERM],
            "Hybrid, lines over 4-30 mm": [HYBRID[0], WIDER, HYBRID[2]], "Size + spiculation + tilt": g["Size"] + g["Spiculation"] + [HYBRID[2]], "Size + radial lines": g["Size"] + g["Radial lines"],
            "Size + surface signs (mask and beam)": g["Size"] + surface_signs_columns, "Surface signs alone (mask and beam)": surface_signs_columns, "Radial lines alone": g["Radial lines"],
            "Spiculation alone": g["Spiculation"], "Size + all designed": g["Size"] + surface_signs_columns + g["Radial lines"]}
    volume = rows[HYBRID[0]].values
    checks = {}
    for c in ("Radial lines: mean", "Radial lines: skin side", "Radial lines: share outward", "Spiculation: SP90", "Spiculation: SPcyl_upper_mean", HYBRID[2], "Classical shape: VHWR", LB_TERM):
        line = np.linalg.lstsq(np.c_[np.ones(len(volume)), volume], rows[c].values, rcond=None)[0]
        checks[c] = dict(direction=auc_either(rows.y.values, rows[c].values)[1], volume_line=line.tolist())
    PLAN.write_text(json.dumps(dict(
        primary=PRIMARY, threshold="p >= 0.5", models={n: fit_model(rows, c) for n, c in sets.items()}, single_features=checks,
        comparisons=[[PRIMARY, "Size + spiculation"], [PRIMARY, "Size"], ["Hybrid + fine budget (LB)", PRIMARY], ["Hybrid, lines over 4-30 mm", PRIMARY], ["Size + spiculation + tilt", "Size + spiculation"]],
        predicted_masks=dict(models=[PRIMARY, "Size"], misses="scored lowest", note="Zhou 2021 masks, three seeds, Test patches centred on the true lesion"),
        already_seen="Every Test result of lb_evaluation, including size + spiculation at AUC 0.889 and ACC 0.871. No Test label or prediction of these descriptors.",
        reading_in_advance={
            "hybrid better than size + spiculation": "only if the 95% interval of the AUC gain excludes zero",
            "LB term useful": "only if the 95% interval of its AUC gain on the hybrid excludes zero",
            "line signal replicated": "if the lower 95% limit of AUC for the mean line excess, in its Train+Validation direction, is above 0.5, after volume as well",
            "reach replicated": "if malignant lesions stay above benign ones in every shell from 4 to 30 mm",
            "sham control passed": "if the 95% interval of lesion minus sham AUC excludes zero"}), indent=1))


def evaluate():
    """One look at Test with the stored models. Directions, volume lines and coefficients all come from Train+Validation."""
    plan = json.loads(PLAN.read_text())
    assert not (OUT / "test_results.csv").exists() or "--force" in sys.argv, "Test has already been scored by these models"
    rows = plan_table()
    keep = (rows.split == "Test").values
    test = rows[keep].reset_index(drop=True)
    y = test.y.values
    assert len(test) == 70
    scores = {n: decision(m, test) for n, m in plan["models"].items()}
    results = {}
    for n, s in scores.items():
        low, high = auc_ci(y, s)
        p = 1 / (1 + np.exp(-s))
        results[n] = dict(challenge.classification(y, p), AUC_low=low, AUC_high=high, sensitivity=(p >= 0.5)[y == 1].mean(), specificity=(p < 0.5)[y == 0].mean())
    pd.DataFrame(results).T.to_csv(OUT / "test_results.csv")
    pd.DataFrame(dict(case=test.case, y=y, **scores)).to_csv(OUT / "test_predictions.csv", index=False)
    challenge.benchmark({n: challenge.classification(y, 1 / (1 + np.exp(-s))) for n, s in scores.items()})[["ACC", "AUC", "Cls_Score", "Cls_Rank"]].to_csv(OUT / "test_benchmark.csv")
    (OUT / "test_comparisons.json").write_text(json.dumps({f"{a} vs {b}": delta(scores[b], scores[a], y) for a, b in plan["comparisons"]}, indent=1))

    single = {}
    for c, check in plan["single_features"].items():
        sign = 1 if check["direction"] == "+" else -1
        raw, residual = sign * test[c].values, sign * (test[c].values - check["volume_line"][0] - check["volume_line"][1] * test[HYBRID[0]].values)
        single[c] = dict(auc=roc_auc_score(y, raw), low=auc_ci(y, raw)[0], high=auc_ci(y, raw)[1], auc_after_volume=roc_auc_score(y, residual), after_low=auc_ci(y, residual)[0], after_high=auc_ci(y, residual)[1], rho_volume=spearmanr(test[c], test[HYBRID[0]])[0])
    pd.DataFrame(single).T.to_csv(OUT / "test_single.csv")

    c, d = np.load(OUT / "control.npz"), np.load(OUT / "features.npz")
    shells = [f"{a:g}-{b:g} mm" for a, b in zip(c["shells"][:-1], c["shells"][1:])]
    pd.DataFrame({(place, "malignant" if label else "benign", s): dict(mean=np.nanmean(values[keep & (c["y"] == label), i]), se=np.nanstd(values[keep & (c["y"] == label), i]) / np.sqrt(np.isfinite(values[keep & (c["y"] == label), i]).sum()))
                  for place, values in (("lesion", c["by_shell"]), ("sham", c["sham_by_shell"])) for label in (1, 0) for i, s in enumerate(shells)}).T.to_csv(OUT / "test_shells.csv")
    room = keep & np.isfinite(c["sham_radial"])
    sham = delta(c["sham_radial"][room], c["radial"][room], c["y"][room])
    (OUT / "test_control.json").write_text(json.dumps(dict(lesions=int(keep.sum()), with_sham=int(room.sum()), lesion_minus_sham=sham, auc_lesion=roc_auc_score(y, c["radial"][keep]), auc_dark=roc_auc_score(y, c["dark_radial"][keep]),
                                                           sham_auc_ci=auc_ci(c["y"][room], c["sham_radial"][room]).tolist(), coupling=dict(malignant=float(np.median(d["coupling"][keep, 0][y == 1])), benign=float(np.median(d["coupling"][keep, 0][y == 0])), auc=roc_auc_score(y, d["coupling"][keep, 0]))), indent=1))

    # The same stored models on Zhou 2021's masks. Patches are centred on the true lesion, so this is still generous.
    from abus_classification.data.lesion_cache import CACHE
    p = np.load(OUT / "predicted.npz")
    cache = np.load(CACHE)
    assert (cache["y"][cache["split"]] == y).all()
    assert spearmanr(np.log(p["volume"][:, 0]), test[HYBRID[0]])[0] > 0.98        # the same lesions in the same order
    columns = descriptor_columns(p)
    frames = [pd.DataFrame({"Size: log volume": np.log(p["volume"][:, r]), "Size: log area": np.log(p["area"][:, r]), **{k: v[:, r] for k, v in columns.items()}}) for r in range(p["volume"].shape[1])]
    out = []
    for name in plan["predicted_masks"]["models"]:
        for r, frame in enumerate(frames):
            s = decision(plan["models"][name], frame)
            misses = ~np.isfinite(s)
            s[misses] = np.nanmin(s) - 1.0
            out.append(dict(model=name, masks="true, on the patch" if r == 0 else f"predicted, seed {r - 1}", AUC=roc_auc_score(y, s), ACC=((s >= 0) == y).mean(), misses=int(misses.sum()), low=auc_ci(y, s)[0], high=auc_ci(y, s)[1]))
    pd.DataFrame(out).to_csv(OUT / "test_predicted.csv", index=False)


if __name__ == "__main__":
    if "--freeze" in sys.argv:
        freeze()
    elif "--evaluate" in sys.argv:
        evaluate()
    elif "--predicted" in sys.argv:
        predicted()
    elif "--control" in sys.argv:
        control()
    elif "--samples" in sys.argv:
        samples()
    elif "--analyse" in sys.argv:
        analyse()
    else:
        main()
