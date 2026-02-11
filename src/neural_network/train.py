"""
Script de antrenare pentru autoencoderul video Conv3D (Etapa 6).

Responsabilități principale:
- încarcă datele preprocesate din `data/train/x_train.npy` și `data/train/x_val.npy`,
- antrenează modelul `ConvLSTMAutoencoder` timp de 100 de epoci,
- salvează cel mai bun model în `models/trained_model.pt`,
- salvează evoluția loss-ului atât ca imagine (`docs/loss_curve.png`),
  cât și ca fișier CSV (`results/training_history.csv`) pentru analiză ulterioară.
"""

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .model import ConvLSTMAutoencoder

ROOT = Path(__file__).parent.parent.parent
TRAIN_PATH = ROOT / "data" / "train" / "x_train.npy"
VAL_PATH = ROOT / "data" / "train" / "x_val.npy"
MODEL_PATH = ROOT / "models" / "trained_model.pt"
PLOT_PATH = ROOT / "docs" / "loss_curve.png"
HISTORY_PATH = ROOT / "results" / "training_history.csv"

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
# Antrenăm explicit 100 de epoci pentru a maximiza capacitatea de învățare,
# conform descrierii din README Etapa 6.
EPOCHS = 100
def _select_device() -> torch.device:
    """
    Selectează device-ul disponibil pentru PyTorch.

    Notă: pe macOS cu Intel + AMD (ex. Radeon Pro 5500M), PyTorch nu poate folosi GPU-ul AMD dedicat.
    CUDA nu este disponibil, iar backend-ul MPS (Metal) este destinat în principal Apple Silicon.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    # ATENȚIE: Conv3D nu este suportat pe MPS în PyTorch în versiunea actuală,
    # deci evităm explicit mps și cădem pe CPU.
    return torch.device("cpu")


DEVICE = _select_device()

def train_model() -> None:
    """
    Rulează bucla de antrenare și persistă modelul + istoricul de loss.

    Notă: pe CPU, scriptul este gândit să ruleze și pe mașini fără GPU,
    folosind câteva optimizări simple (num_threads, mkldnn dacă este disponibil).
    """
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Optimizări CPU (utile mai ales pe macOS/Intel unde nu avem CUDA)
    if DEVICE.type == "cpu":
        try:
            torch.backends.mkldnn.enabled = True
        except Exception:
            pass
        # Folosește toate core-urile disponibile (PyTorch alege default, dar setarea explicită ajută uneori)
        torch.set_num_threads(max(1, (os.cpu_count() or 1)))

    print(f"Training pe device: {DEVICE}")
    if DEVICE.type == "cpu":
        print("Notă: PyTorch pe macOS/Intel nu poate folosi Radeon Pro 5500M. Rulez pe CPU (optimizat).")

    if not TRAIN_PATH.exists() or not VAL_PATH.exists():
        print(f"Datele de antrenare/validare nu au fost găsite în {TRAIN_PATH.parent}.")
        print("Asigură-te că ai rulat mai întâi scriptul de preprocesare / split de date.")
        return

    # Forma de intrare a datelor este (N, T, H, W, C) din pipeline-ul de preprocesare.
    # Pentru Conv3D în PyTorch avem nevoie de (N, C, T, H, W), deci permutăm axele.
    x_train = np.load(TRAIN_PATH)
    x_val = np.load(VAL_PATH)

    train_tensor = torch.FloatTensor(x_train).permute(0, 2, 1, 3, 4)
    val_tensor = torch.FloatTensor(x_val).permute(0, 2, 1, 3, 4)

    # DataLoader: pe CPU ajută paralelizarea; pe MPS/CUDA păstrăm mai conservator.
    num_workers = 0 if DEVICE.type != "cpu" else min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    model = ConvLSTMAutoencoder().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        for batch in train_loader:
            inputs = batch[0].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        
        train_losses.append(t_loss / len(train_loader))

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(DEVICE)
                outputs = model(inputs)
                v_loss += criterion(outputs, inputs).item()
        
        val_losses.append(v_loss / len(val_loader))
        print(f"Epoch {epoch+1}: Train={train_losses[-1]:.6f}, Val={val_losses[-1]:.6f}")

        # Salvăm întotdeauna cel mai bun model pe baza loss-ului de validare,
        # dar nu mai oprim devreme – rulăm toate cele 100 de epoci.
        if val_losses[-1] < best_loss:
            best_loss = val_losses[-1]
            torch.save(model.state_dict(), MODEL_PATH)

    # Salvăm istoricul de loss în format CSV pentru analiza ulterioară (Etapa 6).
    with open(HISTORY_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for idx, (tr, vl) in enumerate(zip(train_losses, val_losses), start=1):
            writer.writerow([idx, tr, vl])
    print(f"Istoricul de antrenare a fost salvat în {HISTORY_PATH}")

    # Generăm și graficul loss-ului, așa cum este menționat în README.
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()

if __name__ == "__main__":
    train_model()