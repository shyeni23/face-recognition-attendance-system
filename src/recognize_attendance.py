import os
import pickle
import cv2
import face_recognition
import pandas as pd
import datetime

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENCODINGS_PATH = os.path.join(BASE_DIR, "encodings", "encodings.pkl")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance")
ATTENDANCE_FILE = os.path.join(ATTENDANCE_DIR, "attendance.csv")

if not os.path.exists(ATTENDANCE_DIR):
    os.makedirs(ATTENDANCE_DIR)

# ---------------- LOAD ENCODINGS ----------------
with open(ENCODINGS_PATH, "rb") as f:
    known_encodings, known_ids = pickle.load(f)

print("[INFO] Encodings loaded successfully")

# ---------------- ATTENDANCE FILE ----------------
if not os.path.exists(ATTENDANCE_FILE):
    df = pd.DataFrame(columns=["ID", "Date", "Time"])
    df.to_csv(ATTENDANCE_FILE, index=False)

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

print("[INFO] Camera started. Press Q to quit.")

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = face_recognition.compare_faces(
            known_encodings, face_encoding, tolerance=0.5
        )
        face_distances = face_recognition.face_distance(
            known_encodings, face_encoding
        )

        name = "Unknown"

        if True in matches:
            best_match_index = face_distances.argmin()
            student_id = known_ids[best_match_index]
            name = f"ID {student_id}"

            now = datetime.datetime.now()
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")

            df = pd.read_csv(ATTENDANCE_FILE)
            if not ((df["ID"] == student_id) & (df["Date"] == date)).any():
                df.loc[len(df)] = [student_id, date, time]
                df.to_csv(ATTENDANCE_FILE, index=False)
                print(f"[ATTENDANCE] Marked for ID {student_id}")

        # Draw box & name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
