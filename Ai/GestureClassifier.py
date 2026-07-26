import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GestureLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_bn = nn.BatchNorm1d(input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * 2  # bidirectional

        self.attn = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_out_dim // 2, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> BatchNorm1d expects (B, D, T)
        x = x.transpose(1, 2)
        x = self.input_bn(x)
        return x.transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, lengths: torch.Tensor = None):
        B, T, _ = x.shape
        x = self._normalize_input(x)

        if lengths is not None:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T)
        else:
            lstm_out, _ = self.lstm(x)

        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=x.device)

        scores = self.attn(lstm_out).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-1e9"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (lstm_out * weights).sum(dim=1)

        logits = self.classifier(pooled)
        return logits


