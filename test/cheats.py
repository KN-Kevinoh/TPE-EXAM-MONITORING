from os import listdir
from numpy import asarray
from keras.models import load_model
import numpy as np
import cv2 
import tensorflow as tf

#0=exchange Paper, 1=looking at friend, 2=Talking Friend, 3=Use Cheat Sheet, 4=No Cheat
labels_cheats = ("exchange Paper" , "looking at friend" , "Talking Friend", "Use Cheat Sheet", "No Cheat")

def image_processing(img):
    # Convert image to grayscale image
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Modify image contrast
    #B, G, R = cv2.split(img)
    #R = cv2.equalizeHist(R.astype(np.uint8))
    #G = cv2.equalizeHist(G.astype(np.uint8))
    #B = cv2.equalizeHist(B.astype(np.uint8))
    
    #img = cv2.merge((B, G, R))
    
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

    #_____END_____#
    return img 


def image_resize2(path, input_size=224, is_color=False):
    img = tf.keras.preprocessing.image.load_img(path, grayscale=not(is_color), color_mode='rgb',
                                                 target_size=(input_size, input_size),
                                                 interpolation='nearest')
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    img = image_processing(img_arr)
    img = np.array(img_arr)
    return img

def image_resize(img, img_size=(299, 299)):
    img = cv2.resize(img,img_size)
    img = asarray(img)
    return img

def pred_activty(path, i = 0, input_size = 224, is_color=True):
  model = load_model("/opt/Documents/projets/models/H_A_R_model_xception.hdf5")
  test_img = image_resize(path, input_size=input_size)
  print(test_img.shape)
  test_img= np.expand_dims(test_img, axis=0)
  test_img = test_img.astype('float32')
  test_img /= 255
  print(test_img.shape)
  pred = model.predict(test_img)

  pred_res = pred[0]
  pred_max = max(pred_res)

  # index
  indice_max = -1
  for i in range(len(pred_res)):
    if pred_res[i] == pred_max:
      indice_max = i
      print(indice_max)
  
  import cv2
  import matplotlib.pyplot as plt
  frame = cv2.imread(path)
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  # draw the predicted activity on the frame
  cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
  cv2.putText(frame, labels_cheats[indice_max], (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 255, 255), 2)
  
  
VIDEO_PATH = "/home/kevin/Videos/Webcam/2020-10-13-091624.webm"
cap = cv2.VideoCapture(VIDEO_PATH)

model = load_model("/opt/Documents/projets/models/H_A_R_model_xception.hdf5")

while cap.isOpened(): # True:
    ret, frame = cap.read()
    frame = image_processing(frame)
    frame = image_resize(frame)
    #print(test_img.shape)
    test_img= np.expand_dims(frame, axis=0)
    test_img = test_img.astype('float32')
    test_img /= 255
    print(test_img.shape)
    pred = model.predict(test_img)

    pred_res = pred[0]
    pred_max = max(pred_res)

    # index
    indice_max = -1
    label = "None"
    for i in range(len(pred_res)):
        if pred_res[i] == pred_max:
            indice_max = i
            label = labels_cheats[indice_max]
    print(indice_max)
    

    cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
    cv2.putText(frame,label , (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 255, 255), 2)


    cv2.imwrite("SAVE_PATH/img_cheats-{}.jpg".format(np.random.randint(0,1000)), frame)
    #cv2.imshow("Cheat demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()