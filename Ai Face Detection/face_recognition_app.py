from imutils.video import VideoStream
from deepface import DeepFace
import numpy as np
import pickle
import time
import cv2
import os

print("[INFO] loading encodings...")
with open("trainer/deep_model.pkl", "rb") as f:
	recognizer = pickle.load(f)
with open("trainer/le.pkl", "rb") as f:
	le = pickle.load(f)

print("[INFO] loading name mappings...")
name_map = {}
with open("faces.txt", "r") as f:
	for line in f:
		if "=" in line:
			idx, name = line.strip().split("=", 1)
			name_map[idx] = name

print("[INFO] starting video stream...")
vs = VideoStream(src=2).start()
vs.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vs.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
time.sleep(2.0)

while True:
	frame = vs.read()

	boxes = []
	encodings = []
	names = []

	try:
		representations = DeepFace.represent(img_path = frame, model_name = "Facenet", enforce_detection=True, detector_backend='opencv')
		
		if representations:
			print(f"[DEBUG] Faces detected: {len(representations)}")
		else:
			print("[DEBUG] No faces detected by DeepFace.represent")

		for rep in representations:
			embedding = rep["embedding"]
			facial_area = rep["facial_area"]
			
			x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
			top, right, bottom, left = y, x + w, y + h, x
			
			boxes.append((top, right, bottom, left))
			encodings.append(embedding)

		for encoding in encodings:
			preds = recognizer.predict_proba(np.array(encoding).reshape(1, -1))[0]
			j = recognizer.classes_[preds.argmax()]
			id_name = le.inverse_transform([j])[0]
			
			display_name = name_map.get(id_name, id_name)
			
			names.append(display_name)
		
		if names:
			print(f"[DEBUG] Recognized names: {names}")
		else:
			print("[DEBUG] No names recognized.")

	except ValueError:
		print("[DEBUG] ValueError: No face detected in frame by DeepFace.represent.")
		pass
	except Exception as e:
		print(f"[DEBUG] Error during face detection/recognition: {e}")
		pass

	for ((top, right, bottom, left), name) in zip(boxes, names):
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()
