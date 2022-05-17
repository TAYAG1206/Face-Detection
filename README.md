#ImageFaceDetection

import cv2
import sys

imagePath = "LaCathedralPic.jpg"
cascPath = "haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascPath)

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.6,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} pogi faces!".format(len(faces)))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)


#VideoFaceDetection

import cv2  
  
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  
  
cap = cv2.VideoCapture(0)

while True:   
    _, img = cap.read()  
   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
  
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)  
   
    for (x, y, w, h) in faces:  
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)  
  
    cv2.imshow('Video', img)  
  
    k = cv2.waitKey(30) & 0xff  
    if k==27:  
        break  
          
cap.release()  
