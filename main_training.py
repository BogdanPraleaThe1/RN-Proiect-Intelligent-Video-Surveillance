"""
Script vechi de antrenare + inferență pentru arhitectura alternativă `AnomalyDetector`.

IMPORTANT:
- Etapa 6 folosește pipeline-ul nou bazat pe `src/neural_network/train.py`,
  `src/neural_network/evaluate.py` și `src/app/main.py`.
- Acest fișier este păstrat doar pentru a documenta iterațiile anterioare și
  nu este folosit în fluxul final descris în README Etapa 6.
"""

import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model_architecture import AnomalyDetector

MODEL_PATH = "anomaly_detector_weights.pth"
DATA_NORMAL_PATH = "data/normal_sequences.npy"
DATA_ABNORMAL_PATH = "data/abnormal_sequences.npy"

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
ANOMALY_THRESHOLD = 0.003


def train_model() -> None:
    """Antrenează modelul `AnomalyDetector` pe un set de date normale (variantă veche)."""
    print("--- Începe Antrenarea Modelului pe Date Normale (variantă veche) ---")
    try:
        normal_data = np.load(DATA_NORMAL_PATH)
        print(f"Dimensiunea totală a datelor normale: {normal_data.shape}")
    except FileNotFoundError:
        print(f"Eroare: Nu s-au găsit datele la {DATA_NORMAL_PATH}. Rulați data_processing.py.")
        return

    normal_tensors = torch.from_numpy(normal_data).float()
    dataset = TensorDataset(normal_tensors, normal_tensors)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = AnomalyDetector()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)

        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel antrenat și salvat la: {MODEL_PATH}")


def calculate_anomaly_score(model: AnomalyDetector, sequence: np.ndarray) -> float:
    """Calculează scorul de anomalie (MSE) pentru o singură secvență."""
    model.eval()
    with torch.no_grad():
        input_tensor = torch.from_numpy(sequence).float()

        if input_tensor.dim() == 5:
            input_batch = input_tensor
        else:
            input_batch = input_tensor.unsqueeze(0)

        reconstructed_tensor = model(input_batch)

        mse_loss = nn.MSELoss()
        score = mse_loss(input_batch, reconstructed_tensor).item()
        return score


def run_inference() -> None:
    """Rulează o inferență simplă pe un exemplu anormal (variantă veche)."""
    print("\n--- Începe Faza de Testare (Inferență) ---")

    model = AnomalyDetector()
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except FileNotFoundError:
        print("Eroare: Modelul nu a fost găsit. Antrenați modelul mai întâi!")
        return

    try:
        abnormal_data = np.load(DATA_ABNORMAL_PATH)
    except FileNotFoundError:
        print("Eroare: Nu s-au găsit datele anormale de test.")
        return

    if abnormal_data.size == 0:
        print("Nu există secvențe anormale de testat.")
        return

    sequence_anormal_test = abnormal_data[0]

    anomaly_score = calculate_anomaly_score(model, sequence_anormal_test)

    print(f"Scorul de Anomalie Calculat: {anomaly_score:.6f}")

    if anomaly_score > ANOMALY_THRESHOLD:
        print("\n🚨 ALERTĂ DETECTATĂ!")

        alert_json = {
            "Camera_ID": "CCTV-Sector-A01",
            "Anomaly_Start_Time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "Anomaly_Score": f"{anomaly_score:.6f}",
            "Suggested_Anomaly_Type": "Comportament Agresiv/Luptă",
            "Threshold_Used": ANOMALY_THRESHOLD,
        }

        print(json.dumps(alert_json, indent=4))
    else:
        print("Comportament normal detectat.")


if __name__ == "__main__":
    train_model()
    run_inference()
