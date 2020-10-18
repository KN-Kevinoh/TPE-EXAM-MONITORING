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
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Modify image contrast
    #B, G, R = cv2.split(img)
    #R = cv2.equalizeHist(R.astype(np.uint8))
    #G = cv2.equalizeHist(G.astype(np.uint8))
    #B = cv2.equalizeHist(B.astype(np.uint8))
    
    #img = cv2.merge((B, G, R))
    cv2.imwrite("/home/kevin/Pictures/Webcam/test/img_original.jpg", img) 
    # Remove noising
    img = cv2.medianBlur(img, 3)
    
    #-----Converting image to LAB Color model----------------------------------- 
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))

    #-----Converting image from LAB Color model to RGB model--------------------
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
   
    cv2.imwrite("/home/kevin/Pictures/Webcam/test/img_new.jpg", img) 
    #img = image_normalize(img)
    return img   


img = cv2.imread("/home/kevin/Pictures/Webcam/2020-10-18-103446.jpg")
im = image_processing(img)