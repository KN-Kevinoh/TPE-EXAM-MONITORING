# -*- coding: utf-8 -*-
import cv2
from mtcnn_detector import mtcnn_model
from image_pre_processing import image_processing
import pre_processing_facenet as facenet
from PIL import Image, ImageDraw
from numpy import load,  asarray, sum, sqrt, multiply
import numpy as np

def l2_normalize(x):
 return x / sqrt(sum(multiply(x, x)))

def euclidian_distance(list_source, cible, treshold = 0.7):
    cible = cible.reshape((cible.shape[0],1))
    cible = l2_normalize(cible)
    #print('cible: ' ,type(cible), len(cible.shape))
    for index, embedding in enumerate(list_source):
        embedding = embedding.reshape((embedding.shape[0],1))
        embedding = l2_normalize(embedding)
        dist = embedding - cible
        dist = sum(multiply(dist,dist))
        #print("embedding",embedding)
        #print("cible",cible)
        #print('embedding: ' ,type(embedding), embedding.shape)
        
        dist = sqrt(dist)
        if dist <= treshold:
            print("distance euclidian verified: ", dist)
            return True, index
        
    print("distance euclidian unverified: ", dist)
    return False, "UNKNOWN"

def findCosineSimilarity(list_source, cible, treshold = 0.09):
     cible = cible.reshape((cible.shape[0],1))
     #print('cible: ' ,type(cible), cible.shape)
     for index, embedding in enumerate(list_source):
        
         embedding = embedding.reshape((embedding.shape[0],1))
         #print('embedding: ' ,type(embedding), embedding.shape)
         a = np.matmul(np.transpose(embedding), cible)
         b = np.sum(np.multiply(embedding, embedding))
         c = np.sum(np.multiply(cible, cible))
         dist = 1 - (a / (np.sqrt(b) * np.sqrt(c)))
         if dist <= treshold:
            print("Cosinus distance verified: ", dist)
            return [True, index]
        
     print("Cosinus distance unverified: ", dist)
     return [False, "UNKNOWN"]
        
    

# load facenet model
model = facenet.facenet_model()

# load face embeddings
data = load("lfw-faces-embeddings.npz")
trainX_embedding, trainy = data['arr_0'], data['arr_1']


#print(trainy)
# 'Obama.mp4'
cap = cv2.VideoCapture('/home/kevin/Videos/Webcam/2020-10-13-090617.webm') 
#address  = "http://192.168.10.65:8080/video"
#cap.open(address)
while cap.isOpened(): # True:
    name = "UNKNOWN"
    ret, frame = cap.read()
    if not ret:
        print('Erreur lecture de la vidéo')
        break
    
    frame = image_processing(frame)
    
    #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    pil_img = Image.fromarray(frame)
    draw  = ImageDraw.Draw(pil_img)
    
    faces_bounding = mtcnn_model(frame)
    for face in faces_bounding:
        x1, y1, width, height = face['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        #extract face
        face_extracted = frame[y1:y2, x1:x2]
        img = Image.fromarray(face_extracted)
        img = facenet.image_resize(img)
        embedding = facenet.face_embedding(model, img)
        results = euclidian_distance(trainX_embedding, embedding)
        print(results)
        if results[0] == True:
            name = str(trainy[results[1]])
            print(name)
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
            
         #draw.rectangle(((left, top), (right, bottom)), outline=(0,0,255))
        draw.rectangle(((x1, y1), (x2, y2)), outline=color)
        
        #draw label with name
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((x1, y2 - text_height - 10), (x2, y2)), fill=(0, 0, 255), outline=(0,0,255))
        draw.text((x1 + 8, y2 -text_height -2), name, fill=(0, 255, 0))
        
    del draw
    cv2.imwrite("SAVE_PATH/img_fr-{}.jpg".format(np.random.randint(0,1000)), asarray(pil_img))
    cv2.imshow('window_frame', asarray(pil_img))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()