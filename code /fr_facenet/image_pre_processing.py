# -*- coding: utf-8 -*-
import cv2
import numpy as np



def resize_image(img, resize_img = (160, 160)):
    cv2.resize(img,resize_img, interpolation=cv2.INTER_LINEAR)
    return img
    
def image_normalize(img):
    norm_img = np.zeros((300, 300))
    norm_img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
    return norm_img
    
   
def image_processing(img):
    # Convert image to grayscale image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Modify image contrast
    R, G, B = cv2.split(img)
    R = cv2.equalizeHist(R)
    G = cv2.equalizeHist(G)
    B = cv2.equalizeHist(B)
    
    img = cv2.merge((R, G, R))
    # Remove noising
    img = cv2.medianBlur(img, 3)
    # Normalize image
    img = image_normalize(img)
    return img   
"""
image = cv2.imread('images/promo24.jpg')

img = image_processing(image)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
for face in faces_detected:
    (x, y, w, h) = face
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1);

cv2.imshow("test",img)
#cv2.imwrite("image_normaliser.png")
"""

