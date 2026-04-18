"""
lstm_model.py
-------------
LSTM model definition in PyTorch.
2 LSTM layers, dropout, linear output head.
Predicts next-day return for all 4 ETFs simultaneously.
"""

import torch
import torch.nn as nn


class ETFForecaster(nn.Module):
    """
    Multi-output LSTM that predicts next-day returns for N ETFs.

    Args:
        input_size:  number of features per timestep (40)
        hidden_size: LSTM hidden units (64)
        num_layers:  stacked LSTM layers (2)
        output_size: number of ETFs to predict (4)
        dropout:     dropout between LSTM layers (0.2)
    """

    def __init__(
        self,
        input_size:  int = 40,
        hidden_size: int = 64,
        num_layers:  int = 2,
        output_size: int = 4,
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,       # input shape: (batch, seq_len, features)
        )

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden_size)
        last_step   = lstm_out[:, -1, :]    # take final timestep
        out         = self.dropout(last_step)
        return self.fc(out)                 # (batch, output_size)


if __name__ == "__main__":
    # Quick sanity check
    model = ETFForecaster()
    dummy = torch.randn(32, 20, 40)   # batch=32, seq=20, features=40
    out   = model(dummy)
    print(f"Model output shape: {out.shape}")   # expect (32, 4)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters:   {total_params:,}")
    print("Model OK.")