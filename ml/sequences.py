"""Turn per-engine sensor histories into fixed-length windows for sequence models."""
import numpy as np


def make_windows(df, sensors, win):
    """Return (X, units) with one window per row of df.

    X has shape (n_rows, win, n_sensors). The window for a row holds that
    engine's last `win` readings up to and including that row (never later ones).
    Early rows with fewer than `win` readings are left-padded with the engine's
    first reading. Rows come out in the same order as df.
    """
    ordered = df.sort_values(["unit", "cycle"]).index.equals(df.index)
    assert ordered, "df must be sorted by unit, then cycle"

    windows, units = [], []
    for unit, g in df.groupby("unit", sort=True):
        arr = g[sensors].to_numpy(dtype=np.float32)
        padded = np.vstack([np.repeat(arr[:1], win - 1, axis=0), arr])
        idx = np.arange(win)[None, :] + np.arange(len(arr))[:, None]
        windows.append(padded[idx])
        units.append(np.full(len(arr), unit))
    return np.concatenate(windows), np.concatenate(units)