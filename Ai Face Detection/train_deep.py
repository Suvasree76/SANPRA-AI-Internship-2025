from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from imutils import paths
from deepface import DeepFace
import pickle
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images("dataset"))

knownEncodings = []
knownNames = []

for (i, imagePath) in enumerate(imagePaths):
	print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)
	try:
		representations = DeepFace.represent(img_path = image, model_name = "Facenet", enforce_detection=True)
	except ValueError as e:
		print(f"Could not detect face in {imagePath}: {e}")
		continue
	for rep in representations:
		embedding = rep["embedding"]
		knownEncodings.append(embedding)
		knownNames.append(name)

print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
le = LabelEncoder()
labels = le.fit_transform(data["names"])

print("[INFO] training model...")
recognizer = KNeighborsClassifier(n_neighbors=5)
recognizer.fit(data["encodings"], labels)

if not os.path.exists('trainer'):
	os.makedirs('trainer')
with open("trainer/deep_model.pkl", "wb") as f:
	pickle.dump(recognizer, f)
with open("trainer/le.pkl", "wb") as f:
	pickle.dump(le, f)

print("[INFO] training complete.")
