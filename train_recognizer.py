import cv2
import os
import numpy as np
from pathlib import Path

DATASET_DIR = Path("dataset")
MODEL_PATH = Path("face_recognizer.yml")
LABELS_PATH = Path("labels.txt")

def load_images_and_labels():
    faces, labels, label_map = [], [], {}
    current_id = 0

    if not DATASET_DIR.exists():
        raise RuntimeError("dataset/ folder not found. Run capture_faces.py first.")

    for person_name in sorted(os.listdir(DATASET_DIR)):
        person_path = DATASET_DIR / person_name
        if not person_path.is_dir():
            continue
        label_map[current_id] = person_name

        for img_name in os.listdir(person_path):
            img_path = person_path / img_name
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(cv2.resize(img, (200, 200)))
            labels.append(current_id)
        current_id += 1

    if len(faces) == 0:
        raise RuntimeError("No images found. Make sure dataset/ has images.")

    return faces, labels, label_map

def main():
    faces, labels, label_map = load_images_and_labels()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.write(str(MODEL_PATH))

    with open(LABELS_PATH, "w") as f:
        for k, v in label_map.items():
            f.write(f"{k}:{v}\n")

    print(f"[INFO] Trained model saved to {MODEL_PATH}")
    print(f"[INFO] Labels saved to {LABELS_PATH}")

if __name__ == "__main__":
    main()
