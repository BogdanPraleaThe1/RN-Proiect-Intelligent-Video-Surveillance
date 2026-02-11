"""
Definirea arhitecturii autoencoderului video Conv3D folosit în Etapa 6.

Input așteptat: tensor de formă (N, 1, T, H, W), unde:
- N: numărul de secvențe din batch,
- 1: canal (imagini grayscale),
- T: numărul de cadre în secvență (ex. 16),
- H, W: înălțime și lățime (ex. 128 x 128).

Modelul comprimă în encoder reprezentarea spațio-temporală și încearcă să
reconstruiască fidel secvența originală în decoder. Eroarea de reconstrucție
este folosită ulterior ca scor de anomalie.
"""

import torch
import torch.nn as nn


class ConvLSTMAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Encoder 3D: două blocuri Conv3D + ReLU + MaxPool pe dimensiunile spațiale.
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(16, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )

        # Decoder 3D: două straturi deconvoluționale care refac rezoluția spațială.
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                8,
                16,
                kernel_size=(1, 2, 2),
                stride=(1, 2, 2),
            ),
            nn.ReLU(),
            nn.ConvTranspose3d(
                16,
                1,
                kernel_size=(1, 2, 2),
                stride=(1, 2, 2),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rulează encoderul + decoderul și întoarce reconstrucția.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x