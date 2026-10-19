import os
import time
import random
import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             average_precision_score, accuracy_score)

RES, PRED = "results/results.csv", "predictions"


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    import torch
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def best_threshold(y, p, grid=np.linspace(0.02, 0.98, 97)):
    f = [f1_score(y, p >= t, zero_division=0) for t in grid]
    i = int(np.argmax(f))
    return float(grid[i]), float(f[i])


def metrics(y, p, t):
    yp = (np.asarray(p) >= t).astype(int)
    return {"f1_sarc": f1_score(y, yp, zero_division=0),
            "precision": precision_score(y, yp, zero_division=0),
            "recall": recall_score(y, yp, zero_division=0),
            "macro_f1": f1_score(y, yp, average="macro"),
            "pr_auc": average_precision_score(y, p),
            "accuracy": accuracy_score(y, yp)}


def report(name, y_val, p_val, y_test=None, p_test=None, *, seed=0, prep="",
           train_time=0.0, n_params=0, extra=None):
    """Threshold chosen on VAL only, then applied to test. Logs + saves predictions."""
    t, _ = best_threshold(y_val, p_val)
    row = {"model": name,
           "prep": prep,
           "seed": seed,
           "threshold": t,
           "train_time_s": round(train_time, 1),
           "n_trained_params": n_params,
           **{f"val_{k}": round(v, 4) for k, v in metrics(y_val, p_val, t).items()}}
    os.makedirs(PRED, exist_ok=True)
    pd.DataFrame({"id": range(len(p_val)), "y_true": y_val, "y_pred": (p_val >= t).astype(int),
                  "prob": p_val}).to_csv(f"{PRED}/{name}_seed{seed}_val.csv", index=False)
    if p_test is not None:
        row.update({f"test_{k}": round(v, 4)
                   for k, v in metrics(y_test, p_test, t).items()})
        pd.DataFrame({"id": range(len(p_test)), "y_true": y_test, "y_pred": (p_test >= t).astype(int),
                      "prob": p_test}).to_csv(f"{PRED}/{name}_seed{seed}.csv", index=False)
    row.update(extra or {})
    os.makedirs("results", exist_ok=True)
    pd.DataFrame([row]).to_csv(
        RES, mode="a", header=not os.path.exists(RES), index=False)
    print(f"[{name} s{seed}] thr={t:.2f} VAL F1={row['val_f1_sarc']:.4f} "
          f"P={row['val_precision']:.3f} R={row['val_recall']:.3f} PR-AUC={row['val_pr_auc']:.4f}")
    return row


class Timer:
    def __enter__(self): self.t = time.time(); return self
    def __exit__(self, *a): self.s = time.time() - self.t
