import cv2
import numpy as np
from PIL import Image
import os

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(BASE_DIR, '..', 'dataset')
cascade_path = os.path.join(
    BASE_DIR, '..', 'haarcascade', 'haarcascade_frontalface_default.xml'
)
trainer_path = os.path.join(BASE_DIR, '..', 'trainer', 'trainer.yml')

# ---------------- LOAD CASCADE ----------------
detector = cv2.CascadeClassifier(cascade_path)
if detector.empty():
    print("ERROR: Haar cascade file not loaded.")
    exit()

# ---------------- CREATE RECOGNIZER ----------------
recognizer = cv2.face.LBPHFaceRecognizer_create()

# ---------------- READ DATASET ----------------
def get_images_and_labels(path):
    face_samples = []
    ids = []

    if not os.path.exists(path):
        print("ERROR: Dataset folder not found.")
        exit()

    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)

        if not os.path.isdir(folder_path):
            continue

        # Expected folder name: user_1, user_2, ...
        try:
            id = int(folder_name.split('_')[1])
        except:
            continue

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)

            pil_img = Image.open(image_path).convert('L')
            img_numpy = np.array(pil_img, 'uint8')

            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)

    return face_samples, ids

print("Reading dataset...")
faces, ids = get_images_and_labels(dataset_path)

# ---------------- VALIDATION ----------------
if len(faces) == 0:
    print("ERROR: No faces found in dataset.")
    print("Make sure you have run capture_faces.py correctly.")
    exit()

# ---------------- TRAIN MODEL ----------------
recognizer.train(faces, np.array(ids))

os.makedirs(os.path.dirname(trainer_path), exist_ok=True)
recognizer.save(trainer_path)

print(f"Model trained successfully with {len(faces)} face samples.")
