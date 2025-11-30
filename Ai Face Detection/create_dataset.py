import cv2
import os

if not os.path.exists('dataset'):
    os.makedirs('dataset')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(2)
cam.set(3, 1280)
cam.set(4, 720)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
count = 0

while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1

        if not os.path.exists(f"dataset/{face_id}"):
            os.makedirs(f"dataset/{face_id}")
        cv2.imwrite(f"dataset/{face_id}/" + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 150:
         break

print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
