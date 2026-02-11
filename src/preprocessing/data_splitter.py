import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

def split_data():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    normal_path = DATA_DIR / "normal_sequences.npy"
    abnormal_path = DATA_DIR / "abnormal_sequences.npy"

    if not normal_path.exists():
        print(f"Fisierul lipseste: {normal_path}")
        return

    normal_data = np.load(normal_path)
    x_train, x_val = train_test_split(normal_data, test_size=0.2, random_state=42)

    np.save(TRAIN_DIR / "x_train.npy", x_train)
    np.save(TRAIN_DIR / "x_val.npy", x_val)

    if abnormal_path.exists():
        abnormal_data = np.load(abnormal_path)
        np.save(TEST_DIR / "normal.npy", x_val)
        np.save(TEST_DIR / "abnormal.npy", abnormal_data)
    
    print("Split finalizat cu succes.")

if __name__ == "__main__":
    split_data()