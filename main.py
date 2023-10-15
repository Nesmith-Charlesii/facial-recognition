import numpy as np
import cv2 as cv 

face_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_faces(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img
    
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    return img

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = detect_faces(frame)

    cv.imshow('Video Face Detection', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()