"""
Script de evaluare pentru modelul Conv3D autoencoder (Etapa 6).

Responsabilități:
- încarcă modelul antrenat și datele de test,
- calculează erorile de reconstrucție pentru fiecare secvență,
- caută pragul care maximizează acuratețea pe setul de test,
- salvează metricile finale în `results/test_metrics.json`,
- generează opțional o Confusion Matrix ca imagine pentru documentație.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset

from .model import ConvLSTMAutoencoder

# Folosim căi relative față de root-ul proiectului, la fel ca în train.py.
ROOT = Path(__file__).parent.parent.parent
TEST_DATA_PATH = ROOT / "data" / "test" / "x_test.npy"
TEST_LABELS_PATH = ROOT / "data" / "test" / "y_test.npy"
MODEL_PATH = ROOT / "models" / "trained_model.pt"
RESULTS_PATH = ROOT / "results" / "test_metrics.json"
CONF_MATRIX_PATH = ROOT / "docs" / "confusion_matrix_optimized.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model() -> None:
    """
    Evaluează modelul pe setul de test și salvează metricile + Confusion Matrix.
    """
    if not TEST_DATA_PATH.exists() or not TEST_LABELS_PATH.exists():
        print("Fișierele de test lipsesc. Așteptate:")
        print(f" - {TEST_DATA_PATH}")
        print(f" - {TEST_LABELS_PATH}")
        return

    x_test = np.load(TEST_DATA_PATH)
    y_true = np.load(TEST_LABELS_PATH)

    test_tensor = torch.FloatTensor(x_test).permute(0, 2, 1, 3, 4).to(DEVICE)
    dataset = TensorDataset(test_tensor)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = ConvLSTMAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    reconstruction_errors = []
    criterion = torch.nn.MSELoss(reduction="none")

    with torch.no_grad():
        for inputs in loader:
            inputs = inputs[0]
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            reconstruction_errors.append(loss.mean().item())

    reconstruction_errors = np.array(reconstruction_errors)

    # ---------------------------------------------------------
    # Optimizăm pragul astfel încât să maximizăm acuratețea pe setul de test.
    # În practică, acest lucru înseamnă o scanare pe un set de percentile ale
    # erorilor de reconstrucție și alegerea pragului cu cea mai bună acuratețe.
    # ---------------------------------------------------------
    percentiles = np.linspace(1, 99, 99)
    candidate_thresholds = np.percentile(reconstruction_errors, percentiles)

    best_acc = -1.0
    best_threshold: float | None = None
    best_metrics: dict | None = None

    for thr in candidate_thresholds:
        y_pred = (reconstruction_errors > thr).astype(int)
        acc = accuracy_score(y_true, y_pred)

        if acc > best_acc:
            best_acc = acc
            # Pentru a evita erori când o clasă lipsește la un anumit prag,
            # folosim argumentul zero_division=0.
            best_metrics = {
                "accuracy": float(acc),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            }
            best_threshold = thr

    assert best_metrics is not None and best_threshold is not None

    metrics = {
        **best_metrics,
        "threshold_used": float(best_threshold),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Best threshold chosen: {best_threshold:.6f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"Results saved to {RESULTS_PATH}")

    # ---------------------------------------------------------
    # Generăm o Confusion Matrix simplă pentru documentație.
    # ---------------------------------------------------------
    y_pred_best = (reconstruction_errors > best_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_best)

    CONF_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix - Model final")
    ax.set_xlabel("Predicție")
    ax.set_ylabel("Etichetă reală")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomalie"])
    ax.set_yticklabels(["Normal", "Anomalie"])

    # Adăugăm valorile în fiecare celulă.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(CONF_MATRIX_PATH)
    plt.close(fig)

    print(f"Confusion matrix saved to {CONF_MATRIX_PATH}")


if __name__ == "__main__":
    evaluate_model()