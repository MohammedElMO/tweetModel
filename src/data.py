import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = "https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/main/ArSarcasm-v2/"
LOCAL = os.path.join(os.path.dirname(__file__), "..", "data")


def _read(name):
    p = os.path.join(LOCAL, name)
    return pd.read_csv(p if os.path.exists(p) else BASE + name)


def load_raw():
    full, test = _read("training_data.csv"), _read("testing_data.csv")
    full["id"], test["id"] = range(len(full)), range(len(test))
    for df in (full, test):                       # "TRUE"/"FALSE" -> 1/0
        df["y"] = df["sarcasm"].astype(str).str.upper().eq("TRUE").astype(int)
    return full, test


def imposed_split(full):                          
    return train_test_split(full, test_size=0.10, random_state=42, stratify=full["sarcasm"])


def dedup_train(train, test, verbose=True):
    """1) drop train tweets that also appear in test  (leakage)
       2) drop tweets that appear with contradictory labels  (annotation noise)
       3) keep one copy of the remaining duplicates"""
    from .preprocess import clean_static

    def key(s):
        return s.map(lambda t: clean_static(t, emojis=False))

    n0 = len(train)
    k_test = set(key(test.tweet))
    train = train.assign(_k=key(train.tweet))
    in_test = train._k.isin(k_test)
    train = train[~in_test]
    contra = train.groupby("_k").y.transform("nunique") > 1
    train = train[~contra].drop_duplicates("_k").drop(columns="_k")
    if verbose:
        print(f"[dedup] {n0} -> {len(train)} | removed (in test): {in_test.sum()} "
              f"| contradictory rows dropped: {contra.sum()}")
    return train


def get_data(dedup=True):
    full, test = load_raw()
    train, val = imposed_split(full)
    if dedup:
        train = dedup_train(train, test)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
