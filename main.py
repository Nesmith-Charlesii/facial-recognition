import numpy as np
import cv2 as cv 

# Load the pre-trained model for face detection - this is a Haar Cascade classifier provided by OpenCV
face_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_faces(img):
    # Convert color space of img to grayscale (required for face detection)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_img, 1.3, 5)
    
    if faces is ():
        print("No face detected")
        return img
    
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    return img

# cv.VideoCapture sets up the webcam for video capture - 0 is the default camera
cam_capture = cv.VideoCapture(0)

tracking = True
while tracking:
    ret, frame = cam_capture.read()
    frame = detect_faces(frame)
    cv.imshow('Video Face Detection', frame)

    keypress = cv.waitKey(1)
    if keypress == ord('q'):
        tracking = False

cam_capture.release()
cv.destroyAllWindows()