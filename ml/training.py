"""Training utilities for the GRU RUL model."""
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.common import CAP, SEED
from ml.models import GRUReg


def scale_windows(X, mean, std):
    """Standardise windows with per-sensor mean/std (arrays of shape (n_features,))."""
    return ((X - mean) / std).astype(np.float32)


@torch.no_grad()
def predict(model, X, batch=1024):
    """Predict RUL in cycles, clipped to [0, CAP]."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        out.append(model(torch.from_numpy(X[i:i + batch])).numpy())
    return np.clip(np.concatenate(out) * CAP, 0, CAP)


def train_gru(X_tr, y_tr, X_va, y_va, hidden=64, lr=1e-3, batch=256,
              epochs=40, patience=6, seed=SEED, on_epoch=None):
    """Train with Adam and MSE; keep the weights from the best validation epoch.

    Targets are divided by CAP so the network works on values between 0 and 1.
    Returns (model, best_epoch, best_val_rmse).
    """
    torch.manual_seed(seed)
    model = GRUReg(n_features=X_tr.shape[2], hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr),
                      torch.from_numpy((y_tr / CAP).astype(np.float32))),
        batch_size=batch, shuffle=True,
    )

    best_rmse, best_state, best_epoch, wait = float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val_rmse = float(np.sqrt(np.mean((predict(model, X_va) - y_va) ** 2)))
        if on_epoch:
            on_epoch(epoch, val_rmse)

        if val_rmse < best_rmse - 1e-4:
            best_rmse, best_epoch, wait = val_rmse, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_epoch, best_rmse