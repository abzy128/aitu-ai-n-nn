from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMAutoencoder(nn.Module):
    def __init__(self, hidden_size: int = 32, latent_size: int = 16) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(x)
        latent = torch.relu(self.to_latent(hidden[-1]))
        decoder_hidden = torch.relu(self.from_latent(latent)).unsqueeze(0)
        decoder_cell = torch.zeros_like(decoder_hidden)
        decoder_input = torch.zeros_like(x)
        decoded, _ = self.decoder(decoder_input, (decoder_hidden, decoder_cell))
        return self.output(decoded)


@dataclass(frozen=True)
class LSTMTrainingConfig:
    epochs: int = 35
    batch_size: int = 64
    learning_rate: float = 1e-3
    seed: int = 42


def train_lstm_autoencoder(
    train_windows: np.ndarray, config: LSTMTrainingConfig = LSTMTrainingConfig()
) -> LSTMAutoencoder:
    torch.manual_seed(config.seed)
    model = LSTMAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    data = torch.as_tensor(train_windows, dtype=torch.float32)
    loader = DataLoader(TensorDataset(data), batch_size=config.batch_size, shuffle=True)

    model.train()
    for _ in range(config.epochs):
        for (batch,) in loader:
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
    return model


def reconstruction_scores(model: LSTMAutoencoder, windows: np.ndarray) -> np.ndarray:
    if len(windows) == 0:
        return np.empty((0,), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(windows, dtype=torch.float32)
        reconstructed = model(x)
        scores = torch.mean((reconstructed - x) ** 2, dim=(1, 2))
    return scores.cpu().numpy()
