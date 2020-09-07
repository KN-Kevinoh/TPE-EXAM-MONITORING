# -*- coding: utf-8 -*-
import cv2
from mtcnn_detector import mtcnn_model
from image_pre_processing import image_processing
import pre_processing_facenet as facenet
from PIL import Image, ImageDraw
from numpy import load,  asarray, sum, sqrt, multiply

def l2_normalize(x):
 return x / sqrt(sum(multiply(x, x)))

def euclidian_distance(list_source, cible, treshold = 0.35):
    cible = l2_normalize(cible)
    for index, embedding in enumerate(list_source):
        embedding = l2_normalize(embedding)
        dist = embedding - cible
        dist = sum(multiply(dist,dist))
        print("embedding",embedding)
        print("cible",embedding)
        
        dist = sqrt(dist)
        if dist <= treshold:
            print("distance euclidian verified: ", dist)
            return True, index
        
    print("distance euclidian unverified: ", dist)
    return False, "UNKNOWN"
        
    

# load facenet model
model = facenet.facenet_model()

# load face embeddings
data = load("/home/kevin/spyder-workspace/EXAM_MONITORING/face_recognition/fr_facenet/model/lfw-faces-embeddings.npz")
trainX_embedding, trainy = data['arr_0'], data['arr_1']

cap = cv2.VideoCapture('Obama.mp4') 

while cap.isOpened(): # True:
    name = "UNKNOWN"
    ret, frame = cap.read()
    if not ret:
        print('Erreur lecture de la vidéo')
        break
    
    frame = image_processing(frame)
    
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
        if results[0] == True:
            name = trainy[results[1]]
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
            
         #draw.rectangle(((left, top), (right, bottom)), outline=(0,0,255))
        draw.rectangle(((x1, y1), (x2, y2)), outline=color)
        
        #draw label with name
        text_width, text_height = draw.textsize(name);
        draw.rectangle(((x1, y2 - text_height - 10), (x2, y2)), fill=(0, 0, 255), outline=(0,0,255))
        draw.text((x1 + 6, y2 -text_height -5), name, fill=(255,255,255,255))
       
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness=2
    del draw
    cv2.imshow('window_frame', asarray(pil_img))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()