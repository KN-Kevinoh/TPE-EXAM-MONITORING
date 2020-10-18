from os import listdir
from numpy import asarray
from keras.models import load_model
import numpy as np
import cv2 
import tensorflow as tf
from mtcnn.mtcnn import MTCNN

#0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
labels = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

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

def image_resize(img, img_size=(48, 48)):
    img = cv2.resize(img,img_size)
    img = asarray(img)
    return img

  
  
VIDEO_PATH = "/home/kevin/Videos/Webcam/2020-10-13-090617.webm"
cap = cv2.VideoCapture(VIDEO_PATH)

model = load_model("/opt/Documents/projets/models/ER_model.h5")
detector = MTCNN()

while cap.isOpened(): # True:
    ret, frame = cap.read()
    # detect faces in the image
    frame = image_processing(frame)
    faces = detector.detect_faces(frame)
    for face in faces:
        x, y, h, w = face['box']
        # gray_face = gray_image[h:w, x:y]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        detected_face = frame[int(y):int(y+h), int(x):int(x+w)] #crop detected face
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
        detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
        
        img_pixels = np.array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels = img_pixels.reshape(img_pixels.shape[0], 48, 48, 1)
        img_pixels = img_pixels.astype('float32')
        
        img_pixels /= 255
        
        predictions = model.predict(img_pixels)
        
        #find max indexed array
        max_index = np.argmax(predictions[0])
        
        emotion = labels[max_index]
        
        cv2.putText(frame, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imwrite("SAVE_PATH/img_emotions-{}.jpg".format(np.random.randint(0,1000)), frame)
        #cv2.imshow("ER demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()