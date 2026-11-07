"""Shared code for the C-MAPSS RUL project."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
CAP = 125
SEED = 42

COLS = (
    ["unit", "cycle"]
    + [f"op{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)


def load_fd(fd="FD001"):
    """Load train/test/RUL files and add the capped RUL label to train."""
    train = pd.read_csv(DATA / f"train_{fd}.txt", sep=r"\s+", header=None, names=COLS)
    test = pd.read_csv(DATA / f"test_{fd}.txt", sep=r"\s+", header=None, names=COLS)
    rul = pd.read_csv(DATA / f"RUL_{fd}.txt", header=None, names=["RUL"])
    last = train.groupby("unit")["cycle"].transform("max")
    train["RUL"] = (last - train["cycle"]).clip(upper=CAP)
    return train, test, rul


def select_sensors(train, threshold=0.01):
    """Keep sensors whose standard deviation is at least the threshold."""
    sensors = [c for c in COLS if c.startswith("s")]
    std = train[sensors].std()
    return [s for s in sensors if std[s] >= threshold]


def slope_fn(x):
    """Least-squares slope of a window against time."""
    t = np.arange(len(x)) - (len(x) - 1) / 2
    return (t * (x - x.mean())).sum() / (t ** 2).sum()


def add_features(df, sensors, win, use_slope=False):
    """Rolling mean/std (and optional slope) per engine. Looks backwards only."""
    out = df.copy()
    g = df.groupby("unit")
    for s in sensors:
        out[f"{s}_mean"] = g[s].transform(lambda x: x.rolling(win, min_periods=1).mean())
        out[f"{s}_std"] = g[s].transform(lambda x: x.rolling(win, min_periods=1).std()).fillna(0.0)
        if use_slope:
            out[f"{s}_slope"] = g[s].transform(
                lambda x: x.rolling(win, min_periods=2).apply(slope_fn, raw=True)
            ).fillna(0.0)
    return out


def feature_names(sensors, use_slope=False):
    names = sensors + [f"{s}_mean" for s in sensors] + [f"{s}_std" for s in sensors]
    if use_slope:
        names += [f"{s}_slope" for s in sensors]
    return names


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def nasa_score(y_true, y_pred):
    """Asymmetric score: late predictions are penalised more. Lower is better."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))