import cv2
import numpy as np
import face_recognition

imgTrain = face_recognition.load_image_file('img/sundar_pichai_train.png')
imgTrain = cv2.cvtColor(imgTrain, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('img/sundar_pichai_test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgTrain)[0]
print(faceLoc)

# cv2.imshow('Train Image', imgTrain)
# cv2.imshow('Test Image', imgTest)
# cv2.waitKey(0)



