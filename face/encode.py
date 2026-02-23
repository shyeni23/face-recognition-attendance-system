import face_recognition
import os
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset_dir = os.path.join(BASE_DIR, "dataset")
encodings_dir = os.path.join(BASE_DIR, "encodings")
os.makedirs(encodings_dir, exist_ok=True)

encodings = []
ids = []

print("Starting face encoding...")

for student_folder in os.listdir(dataset_dir):
    student_id = student_folder.split("_")[1]
    folder_path = os.path.join(dataset_dir, student_folder)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        if face_encodings:
            encodings.append(face_encodings[0])
            ids.append(student_id)

with open(os.path.join(encodings_dir, "encodings.pkl"), "wb") as f:
    pickle.dump((encodings, ids), f)

print("Encoding completed successfully.")
