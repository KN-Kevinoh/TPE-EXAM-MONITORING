# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
""" Dataset pre-proccessing
    importation of libraries
"""
import os
#import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

#data_path = "datasets/"


def apply_brightness_contrast(input_img, brightness = 0, contrast = 0, resize_image_width = 1080, resize_image_height = 1080):
    input_img = cv.resize(input_img, (resize_image_height,resize_image_width), 0, 0, cv.INTER_AREA)
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def pre_processing_image(img):
    img = cv.imread(img,1)
    img = apply_brightness_contrast(img,0, 2, img.shape[1] + 100, img.shape[0] + 200)
    
    cv.imwrite("img_eq.jpg", img)
    return "img_eq.jpg"
    
def adjust_gamma(image, gamma=1.0):
   table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
   # apply gamma correction using the lookup table
   return cv.LUT(image, table)


#resize_imge = 800
def frame_image(img):
    #img = cv.equalizeHist(img)
   # img = np.hstack((img,equ)) 
    hist,bins = np.histogram(img.flatten(),256,[0,256])

    cdf = hist.cumsum()
   
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    
    img = cdf[img]
    #frame = apply_brightness_contrast(img,-50,4, img.shape[0], img.shape[1])
    return img
   
#load folder
def loadDatasets(path):
    
    image_files = sorted([os.path.join(path, 'train', file)
         for file in os.listdir(path + "/train") if file.endswith('.jpg')])
 
    return image_files
  
# Display one image
def display_one(img, title_img = "Original"):
    plt.imshow(img), plt.title(title_img)
    plt.xticks([]), plt.yticks([])
    plt.show()
    
# Display two images
def display(img1, img2, title_img1 = "Original", title_img2 = "Edited"):
    plt.subplot(121), plt.imshow(img1), plt.title(title_img1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img2), plt.title(title_img2)
    plt.xticks([]), plt.yticks([])
    plt.show()   
   
# Preprocessing
def processing_multi_images(data):
    # loading image
    img = [cv.imread(i, cv.IMREAD_UNCHANGED) for i in data[:]] 
    print(img)
    # --------------------------------
    # setting dim of the resize

    height = 250
    width = 250
    dim = (width, height)
    res_img = []
    for i in range(len(img)):
        res = cv.resize(img[i], dim, interpolation=cv.INTER_AREA)
        res_img.append(res)

    # Checcking the size
    print("RESIZED", res_img[1].shape)
    
    # Visualizing one of the images in the array
    original = res_img[1]
    display_one(original)
  


