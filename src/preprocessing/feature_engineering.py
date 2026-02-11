import cv2
import numpy as np
import os
import torch

TARGET_SIZE = (128, 128)
SEQ_LENGTH = 16

def preprocess_frame(frame, target_size=TARGET_SIZE):
    if frame is None:
        return None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0
    frame = np.expand_dims(frame, axis=0) 
    
    return frame

def process_single_video(video_path, seq_length=SEQ_LENGTH, target_size=TARGET_SIZE):
    """Procesează un singur fișier video în secvențe."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = preprocess_frame(frame, target_size)
        if processed_frame is not None:
            frames.append(processed_frame)
        
    cap.release()

    sequences = []
    for i in range(0, len(frames) - seq_length + 1, seq_length):
        sequence = np.stack(frames[i:i + seq_length], axis=0) 
        sequences.append(sequence)
        
    return sequences

def create_sequences_from_folder(folder_path, output_filename):
    """Parcurge un folder, procesează toate videoclipurile și salvează rezultatul."""
    
    print(f"\n--- Procesare director: {folder_path} ---")
    all_sequences = []
    
    # Parcurge directorul pentru a găsi toate fișierele (os.walk)
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Filtrează doar fișierele video comune
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(root, file)
                print(f"  Procesare fișier: {file}...")
                
                sequences = process_single_video(video_path, SEQ_LENGTH)
                all_sequences.extend(sequences)
                print(f"  Secvențe generate din {file}: {len(sequences)}")
    
    if all_sequences:
        final_sequences = np.array(all_sequences, dtype=np.float32)
        print(f"--- Total secvențe generate: {len(final_sequences)} ---")
        np.save(output_filename, final_sequences)
        print(f"Datele salvate în: {output_filename}")
        return final_sequences
    else:
        print("Avertisment: Nu s-au găsit fișiere video în director.")
        return np.array([])

if __name__ == '__main__':
    # Definește căile (corespunzătoare structurii tale de foldere)
    FOLDER_NORMAL = 'data/dataset_normal'  
    FOLDER_ANORMAL = 'data/dataset_anormal'
    
    # Asigură-te că directorul 'data' există
    if not os.path.exists('data'):
        os.makedirs('data')

    # 1. Procesează și Salvează datele normale (pentru antrenare)
    create_sequences_from_folder(FOLDER_NORMAL, 'data/normal_sequences.npy')

    # 2. Procesează și Salvează datele anormale (pentru testare)
    create_sequences_from_folder(FOLDER_ANORMAL, 'data/abnormal_sequences.npy')
