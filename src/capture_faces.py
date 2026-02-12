import cv2
import os
import time

# Ask for student ID
student_id = input("Enter Student ID: ")



# Create folder for student
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(BASE_DIR, "dataset", f"student_{student_id}")
os.makedirs(dataset_path, exist_ok=True)

# Open camera
cam = cv2.VideoCapture(0)

total_images = 30
count = 0
last_capture_time = time.time()

print("Camera started.")
print("Slowly move your face left, right, up, down.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to access camera")
        break

    # Show counter on screen
    cv2.putText(
        frame,
        f"Images: {count}/{total_images}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Face Capture", frame)

    # Capture image every 2 seconds
    if time.time() - last_capture_time >= 0.5 and count < total_images:
        img_path = os.path.join(dataset_path, f"{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Captured image {count}")
        count += 1
        last_capture_time = time.time()

    # Exit conditions
    if count >= total_images:
        print("Face capture completed.")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
