"""
Arhitectură alternativă (CNN + LSTM) folosită experimental în etapele inițiale.

NOTĂ IMPORTANTĂ:
 - Etapa 6 folosește ca arhitectură finală autoencoderul Conv3D definit în
   `src/neural_network/model.py` (vezi README Etapa 6).
 - Acest fișier este păstrat doar ca referință istorică / explorare și
   **nu mai este folosit în pipeline-ul principal**.
"""

import torch
import torch.nn as nn

CHANNELS = 1
HEIGHT = 128
WIDTH = 128
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2


class AnomalyDetector(nn.Module):
    def __init__(self, seq_length: int = 16) -> None:
        super().__init__()
        self.seq_length = seq_length

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.feature_size = 64 * 32 * 32

        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_LAYERS,
            batch_first=True,
        )

        self.decoder_linear = nn.Linear(LSTM_HIDDEN_SIZE, self.feature_size)

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, CHANNELS, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, C, H, W = x.size()

        cnn_features = []
        for t in range(seq_length):
            frame_features = self.encoder_cnn(x[:, t, :, :, :])
            cnn_features.append(frame_features.view(batch_size, -1))

        cnn_features = torch.stack(cnn_features, dim=1)

        lstm_output, _ = self.lstm(cnn_features)

        decoded_output = self.decoder_linear(lstm_output)

        reconstructed_frames = []
        for t in range(seq_length):
            frame_tensor = decoded_output[:, t, :].view(batch_size, 64, 32, 32)
            reconstructed_frame = self.decoder_cnn(frame_tensor)
            reconstructed_frames.append(reconstructed_frame)

        return torch.stack(reconstructed_frames, dim=1)
