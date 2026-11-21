"""Sequence models for RUL prediction."""
import torch.nn as nn


class GRUReg(nn.Module):
    """GRU over a window of sensor readings, then a linear layer to predict RUL."""

    def __init__(self, n_features, hidden=64, layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                 # x: (batch, window, n_features)
        out, _ = self.gru(x)              # out: (batch, window, hidden)
        return self.head(out[:, -1]).squeeze(-1)   # use the last time step