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

tracking = True
while tracking:
    ret, frame = cap.read()
    frame = detect_faces(frame)
    cv.imshow('Video Face Detection', frame)
    
    keypress = cv.waitKey(1)
    if keypress == ord('q'):
        tracking = False

cap.release()
cv.destroyAllWindows()