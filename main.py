
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import pickle
import numpy as np
import mediapipe as mp
import time

# ================= LOAD MODEL =================
with open("cognitive_load_model.pkl", "rb") as f:
    model = pickle.load(f)

LABELS = ["LOW", "MEDIUM", "HIGH"]

# ================= MEDIAPIPE =================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.21
WINDOW_SIZE = 30        # frames (~1 second)
PRED_INTERVAL = 1.0     # seconds

# ================= HELPERS =================
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# ================= CAMERA (LIBCAMERA) =================
pipeline = (
    "libcamerasrc ! "
    "video/x-raw,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! "
    "appsink"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ Camera open nahi hua (libcamera)")
    exit()

# ================= STATE =================
ear_values = []
motion_values = []
blink_count = 0
closed_frames = 0
prev_landmarks = None

last_pred = None
last_conf = 0.0
last_update_time = time.time()

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame read fail")
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks])

        left_eye = pts[LEFT_EYE]
        right_eye = pts[RIGHT_EYE]

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
        ear_values.append(ear)

        # Blink detection
        if ear < EAR_THRESHOLD:
            closed_frames += 1
        else:
            if closed_frames >= 2:
                blink_count += 1
            closed_frames = 0

        # Facial motion energy
        if prev_landmarks is not None:
            motion_values.append(np.linalg.norm(pts - prev_landmarks))

        prev_landmarks = pts

    # ================= PREDICTION =================
    current_time = time.time()

    if (
        len(ear_values) >= WINDOW_SIZE
        and (current_time - last_update_time) >= PRED_INTERVAL
    ):
        avg_ear = np.mean(ear_values)
        motion_energy = np.mean(motion_values) if motion_values else 0

        X = np.array([[avg_ear, blink_count, motion_energy]])
        last_pred = model.predict(X)[0]
        last_conf = model.predict_proba(X).max()

        ear_values.clear()
        motion_values.clear()
        blink_count = 0
        closed_frames = 0
        last_update_time = current_time

    # ================= DISPLAY =================
    if last_pred is not None:
        cv2.putText(
            frame,
            f"Cognitive Load: {LABELS[last_pred]} ({last_conf:.2f})",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Real-Time Cognitive Load Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ================= CLEANUP =================
cap.release()
cv2.destroyAllWindows()
