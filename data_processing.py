"""
Script de preprocesare video (variantă simplă, la nivel de proiect).

Funcționalitate:
 - citește clipuri video din directoare,
 - convertește frame-urile la grayscale, redimensionează la 128x128,
 - normalizează în [0, 1] și le grupează în secvențe de lungime 16,
 - salvează rezultatul ca fișiere `.npy` în directorul `data/`.

NOTĂ:
 - Pipeline-ul folosit în Etapa 6 este centrat în `src/preprocessing/`,
   dar acest fișier este păstrat pentru rulări rapide / experimente.
"""

import os

import cv2
import numpy as np

TARGET_SIZE = (128, 128)
SEQ_LENGTH = 16


def preprocess_frame(frame, target_size: tuple[int, int] = TARGET_SIZE) -> np.ndarray | None:
    """Convertim un frame BGR la grayscale, redimensionăm și normalizăm."""
    if frame is None:
        return None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0
    frame = np.expand_dims(frame, axis=0)  # (1, H, W) – canalul la început

    return frame


def process_single_video(
    video_path: str,
    seq_length: int = SEQ_LENGTH,
    target_size: tuple[int, int] = TARGET_SIZE,
) -> list[np.ndarray]:
    """Procesează un singur fișier video în secvențe de lungime `seq_length`."""
    cap = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = preprocess_frame(frame, target_size)
        if processed_frame is not None:
            frames.append(processed_frame)

    cap.release()

    sequences: list[np.ndarray] = []
    for i in range(0, len(frames) - seq_length + 1, seq_length):
        sequence = np.stack(frames[i : i + seq_length], axis=0)  # (T, 1, H, W)
        sequences.append(sequence)

    return sequences


def create_sequences_from_folder(folder_path: str, output_filename: str) -> np.ndarray:
    """Parcurge un folder, procesează toate videoclipurile și salvează rezultatul."""

    print(f"\n--- Procesare director: {folder_path} ---")
    all_sequences: list[np.ndarray] = []

    # Parcurge directorul pentru a găsi toate fișierele (os.walk)
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Filtrează doar fișierele video comune
            if file.endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_path = os.path.join(root, file)
                print(f"  Procesare fișier: {file}...")

                sequences = process_single_video(video_path, SEQ_LENGTH)
                all_sequences.extend(sequences)
                print(f"  Secvențe generate din {file}: {len(sequences)}")

    if all_sequences:
        final_sequences = np.array(all_sequences, dtype=np.float32)
        print(f"--- Total secvențe generate: {len(final_sequences)} ---")
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        np.save(output_filename, final_sequences)
        print(f"Datele salvate în: {output_filename}")
        return final_sequences
    else:
        print("Avertisment: Nu s-au găsit fișiere video în director.")
        return np.array([])


if __name__ == "__main__":
    # Definește căile (corespunzătoare structurii tale de foldere)
    FOLDER_NORMAL = "data/dataset_normal"
    FOLDER_ANORMAL = "data/dataset_anormal"

    # Asigură-te că directorul 'data' există
    if not os.path.exists("data"):
        os.makedirs("data")

    # 1. Procesează și Salvează datele normale (pentru antrenare)
    create_sequences_from_folder(FOLDER_NORMAL, "data/normal_sequences.npy")

    # 2. Procesează și Salvează datele anormale (pentru testare)
    create_sequences_from_folder(FOLDER_ANORMAL, "data/abnormal_sequences.npy")
