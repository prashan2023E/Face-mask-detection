# ================================
# SUPPRESS TensorFlow oneDNN WARNING
# ================================
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ================================
# IMPORT LIBRARIES
# ================================
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ================================
# PATHS (CHANGE ONLY IF NEEDED)
# ================================
BASE_DIR = r"C:\Users\ASUA\facemask"

PROTOTXT_PATH = os.path.join(BASE_DIR, "deploy.prototxt")
MODEL_PATH = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
MASK_MODEL_PATH = os.path.join(BASE_DIR, "mask_cnn_model.h5")

import os

files = [
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
]

for f in files:
    print(f, "EXISTS" if os.path.exists(f) else "MISSING")

# ================================
# LOAD FACE DETECTOR
# ================================
if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Face detector files not found. Check paths.")

face_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

# ================================
# LOAD MASK CNN MODEL
# ================================
mask_model = load_model(MASK_MODEL_PATH)

IMG_SIZE = 128

# ================================
# START WEBCAM
# ================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

print("Real-time Mask Detection started... Press 'q' to exit")

# ================================
# REAL-TIME LOOP
# ================================
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face = face.astype("float32") / 255.0
            face = np.expand_dims(face, axis=0)

            prediction = mask_model.predict(face, verbose=0)[0][0]

            prediction = mask_model.predict(face, verbose=0)[0][0]

            if prediction < 0.5:
                confidence = int((1 - prediction) * 100)
                label = f"MASKED : {confidence}%"
                color = (0, 255, 0)
            else:
                confidence = int(prediction * 100)
                label = f"NO MASK : {confidence}%"
                color = (0, 0, 255)


            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

    cv2.imshow("Real-Time Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================================
# CLEANUP
# ================================
cap.release()
cv2.destroyAllWindows()
