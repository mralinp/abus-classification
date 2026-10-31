"""Evaluation protocol shared by every reproduced paper.

Train on Train+Validation (130 volumes), report on the official Test split (70), malignant = positive class.
"""
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def metrics(y, p):
    pred = p >= 0.5
    return dict(AUC=roc_auc_score(y, p), Acc=accuracy_score(y, pred), Prec=precision_score(y, pred),
                Sens=recall_score(y, pred), Spec=recall_score(y, pred, pos_label=0), F1=f1_score(y, pred))


def folds(y, tv, seed):
    return [(tv[a], tv[b]) for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(tv, y[tv])]


def auc_ci(y, p, n=1000, seed=0):
    rng, aucs = np.random.default_rng(seed), []
    while len(aucs) < n:
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y):
            aucs.append(roc_auc_score(y[i], p[i]))
    return np.percentile(aucs, [2.5, 97.5])


def _claim_gpu(queue):
    if torch.cuda.is_available():
        torch.cuda.set_device(queue.get())


def run_on_gpus(fn, jobs):
    """fn(job) for every job, one process per GPU at a time. fn must be importable; the caller stays off the GPU."""
    ctx = multiprocessing.get_context("spawn")
    workers = max(1, torch.cuda.device_count())
    with ctx.Manager() as manager:
        queue = manager.Queue()
        for gpu in range(workers):
            queue.put(gpu)
        with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_claim_gpu, initargs=(queue,)) as pool:
            return list(pool.map(fn, jobs))
