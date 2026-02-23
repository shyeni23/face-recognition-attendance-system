import os
import pickle
import cv2
import face_recognition
import pandas as pd
import datetime
import numpy as np

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENCODINGS_PATH = os.path.join(BASE_DIR, "encodings", "encodings.pkl")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance")
ATTENDANCE_FILE = os.path.join(ATTENDANCE_DIR, "attendance.csv")

os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# ================= LOAD ENCODINGS =================
with open(ENCODINGS_PATH, "rb") as f:
    known_encodings, known_ids = pickle.load(f)

print("[INFO] Encodings loaded")

# ================= ATTENDANCE FILE =================
if not os.path.exists(ATTENDANCE_FILE):
    df = pd.DataFrame(columns=["ID", "Date", "Time"])
    df.to_csv(ATTENDANCE_FILE, index=False)

# ================= EAR (BLINK) FUNCTIONS =================
def eye_aspect_ratio(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C)

EAR_THRESHOLD = 0.20
REQUIRED_BLINKS = 2

# ================= BLUE LIGHT / SCREEN DETECTION =================
def is_screen_attack(face_region):
    if face_region.size == 0:
        return False

    b, g, r = cv2.split(face_region)

    mean_b = np.mean(b)
    mean_g = np.mean(g)
    mean_r = np.mean(r)

    # Heuristic: phone screens emit stronger blue light
    if mean_b > 1.25 * mean_r and mean_b > 1.15 * mean_g:
        return True
    return False

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
print("[INFO] Blink twice to mark attendance")

blink_counter = 0
blinked = False
attendance_marked = set()  # store IDs already marked in this run

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)
    face_landmarks = face_recognition.face_landmarks(rgb)

    for (top, right, bottom, left), face_encoding, landmarks in zip(
        face_locations, face_encodings, face_landmarks
    ):
        matches = face_recognition.compare_faces(
            known_encodings, face_encoding, tolerance=0.5
        )
        face_distances = face_recognition.face_distance(
            known_encodings, face_encoding
        )

        name = "Unknown"

        if True in matches:
            best_match_index = np.argmin(face_distances)
            student_id = known_ids[best_match_index]
            name = f"ID {student_id}"

            # ---------- SCREEN ATTACK CHECK ----------
            face_roi = frame[top:bottom, left:right]
            if is_screen_attack(face_roi):
                cv2.putText(frame, "SCREEN ATTACK DETECTED",
                            (left, top - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
                continue

            # ---------- BLINK DETECTION ----------
            left_eye = landmarks["left_eye"]
            right_eye = landmarks["right_eye"]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD and not blinked:
                blink_counter += 1
                blinked = True

            if avg_ear >= EAR_THRESHOLD:
                blinked = False

            cv2.putText(frame, f"Blinks: {blink_counter}",
                        (left, bottom + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 0, 0), 2)

            # ---------- MARK ATTENDANCE ----------
            if blink_counter >= REQUIRED_BLINKS and student_id not in attendance_marked:
                now = datetime.datetime.now()
                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")

                df = pd.read_csv(ATTENDANCE_FILE)
                if not ((df["ID"] == student_id) & (df["Date"] == date)).any():
                    df.loc[len(df)] = [student_id, date, time]
                    df.to_csv(ATTENDANCE_FILE, index=False)
                    print(f"[ATTENDANCE] Marked for ID {student_id}")

                attendance_marked.add(student_id)

        # ---------- DRAW FACE BOX ----------
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
