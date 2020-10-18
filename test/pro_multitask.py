from threading import Thread

import cv2
from keras.models import load_model
import pre_processing_facenet as facenet
from mtcnn.mtcnn import MTCNN
from pro_detected import MarkDetector

from PIL import Image, ImageDraw

import numpy as np 


from os import listdir
from numpy import asarray
import numpy as np 
import tensorflow as tf
import dlib
from sklearn.preprocessing import StandardScaler
from numpy import load,  asarray, sum, sqrt, multiply
import math


######### implements thread
import sys
from builtins import super    # https://stackoverflow.com/a/30159479

if sys.version_info >= (3, 0):
    _thread_target_key = '_target'
    _thread_args_key = '_args'
    _thread_kwargs_key = '_kwargs'
else:
    _thread_target_key = '_Thread__target'
    _thread_args_key = '_Thread__args'
    _thread_kwargs_key = '_Thread__kwargs'

class ThreadSchedule(Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._return = None

    def run(self):
        target = getattr(self, _thread_target_key)
        if not target is None:
            self._return = target(
                *getattr(self, _thread_args_key),
                **getattr(self, _thread_kwargs_key)
            )

    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        return self._return


############ Preprocessing function ############

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
 
 
########### heads

# compute face landmarks distance      
def compute_features(face_points):
    assert (len(face_points) == 68), "len(face_points) must be 68"
    
    face_points = np.array(face_points)
    features = []
    for i in range(68):
        for j in range(i+1, 68):
            features.append(np.linalg.norm(face_points[i]-face_points[j]))
            
    return np.array(features).reshape(1, -1)


def label_head(cible):
   
    if cible < 0:
        return "left"
    else:
        return "right"
  

model1 = load_model("models/head_pose_model.h5")

# head pose train
data= np.load("datasets/head-train.npz")
data_train = data['arr_0']
std = StandardScaler()
std.fit(data_train)

def model_head(model ,marks):
            
    features = compute_features(marks)
   
    features = std.transform(features)
    
    y_pred = model.predict(features)

    roll_pred, pitch_pred, yaw_pred = y_pred[0]
    # Get the absolute value for each number
    absValues = [math.fabs(number) for number in y_pred[0]]
    absValues = np.array(absValues)
    max_val = max(absValues)
    print(roll_pred, pitch_pred, yaw_pred)
    label = "Normal"
    if max_val == math.fabs(roll_pred): 
        label = label_head(roll_pred)
    
    if max_val == math.fabs(yaw_pred): 
        label = label_head(yaw_pred)
        
    return label

######### fin head


########## fr debut ###############

# load facenet model
model2 = facenet.facenet_model()

# load face embeddings
data = load("lfw-faces-embeddings.npz")
trainX_embedding, trainy = data['arr_0'], data['arr_1']


def l2_normalize(x):
 return x / sqrt(sum(multiply(x, x)))

def euclidian_distance(list_source, cible, treshold = 0.9):
    cible = cible.reshape((cible.shape[0],1))
    cible = l2_normalize(cible)
   
    for index, embedding in enumerate(list_source):
        embedding = embedding.reshape((embedding.shape[0],1))
        embedding = l2_normalize(embedding)
        dist = embedding - cible
        dist = sum(multiply(dist,dist))
       
        dist = sqrt(dist)
        if dist <= treshold:
            print("distance euclidian verified: ", dist)
            return True, index
        
    print("distance euclidian unverified: ", dist)
    return False, "UNKNOWN"

def findCosineSimilarity(list_source, cible, treshold = 0.09):
     cible = cible.reshape((cible.shape[0],1))
     
     for index, embedding in enumerate(list_source):
        
         embedding = embedding.reshape((embedding.shape[0],1))
        
         a = np.matmul(np.transpose(embedding), cible)
         b = np.sum(np.multiply(embedding, embedding))
         c = np.sum(np.multiply(cible, cible))
         dist = 1 - (a / (np.sqrt(b) * np.sqrt(c)))
         if dist <= treshold:
            print("Cosinus distance verified: ", dist)
            return [True, index]
        
     print("Cosinus distance unverified: ", dist)
     return [False, "UNKNOWN"]
        


def model_fr(model, face):

    name = "UNKNOWN"
    img = Image.fromarray(face)
    img = facenet.image_resize(img)
    embedding = facenet.face_embedding(model, img)
    results = euclidian_distance(trainX_embedding, embedding)
    #print(results)
    if results[0] == True:
        name = str(trainy[results[1]])
        #print(name)
    
    return name

############ fin fr #######################""


####### debut emotions ###########

#0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
labels_emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

model3 = load_model("models/ER_model.h5")

def model_ER(model, face):
    
    detected_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) #transform to gray scale
    detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
    
    img_pixels = np.array(detected_face)
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    img_pixels = img_pixels.reshape(img_pixels.shape[0], 48, 48, 1)
    img_pixels = img_pixels.astype('float32')
    
    img_pixels /= 255
    
    predictions = model.predict(img_pixels)
    
    #find max indexed array
    max_index = np.argmax(predictions[0])
    
    emotion = labels_emotions[max_index]
   
    return emotion

##### emotion fin ############

######## Cheat debut #######################

#0=exchange Paper, 1=looking at friend, 2=Talking Friend, 3=Use Cheat Sheet, 4=No Cheat
labels_cheats = ("exchange Paper" , "looking at friend" , "Talking Friend", "Use Cheat Sheet", "No Cheat")

model4 = load_model("models/H_A_R_model_xception.hdf5")

def image_resize(img, img_size=(299, 299)):
    img = cv2.resize(img,img_size)
    img = asarray(img)
    return img


def model_cheat(model, frame):
    frame = image_resize(frame)
    #print(test_img.shape)
    test_img= np.expand_dims(frame, axis=0)
    test_img = test_img.astype('float32')
    test_img /= 255
    #print(test_img.shape)
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
    #print(indice_max)
    
    return label

########### fin cheat ##########################

################################ MTASK #########%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


VIDEO_PATH = '/home/kevin/spyder-workspace/videos_pris/test2.mp4'
vs = cv2.VideoCapture(VIDEO_PATH)

#src = 0
#vs = cv2.VideoCapture(src)

#address  = "http://ip:8080/video"
#vs.open(address)
i = 0
mark_detector = MarkDetector()
while vs.isOpened():
    _bool, frame = vs.read()
    # frame count
    i = i + 1
    
    if not _bool:
        print('Erreur lecture de la vidéo')
        break
    
    frame = image_processing(frame)
    #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    faceboxes = mark_detector.extract_cnn_facebox(frame)
    for facebox in faceboxes:
        face_img = frame[facebox[1]: facebox[3],
                facebox[0]: facebox[2]]
        face_img = cv2.resize(face_img, (128, 128))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        marks = mark_detector.detect_marks([face_img])
        marks *= (facebox[2] - facebox[0])
        marks[:, 0] += facebox[0]
        marks[:, 1] += facebox[1]
        shape = marks.astype(np.uint)
        print(marks.shape)
        #mark_detector.draw_marks(frame, marks, color=(0, 255, 0))
    
  
        t1 = ThreadSchedule(target=model_head, args=(model1, marks))
        t2 = ThreadSchedule(target=model_fr, args=(model2, face_img))
        t3 = ThreadSchedule(target=model_ER, args=(model3, face_img))
        t4 = ThreadSchedule(target=model_cheat, args=(model4, face_img))
        
        # Threads started
        t1.start()
        t2.start()
        t3.start()
        t4.start()
        
        # waiting threads results
        results_1 = t1.join()
        results_2 = t2.join()
        results_3 = t3.join()
        results_4 = t4.join()
    
        print("############ Debut", i," ième Frame(s) ###############")
        print(results_1)
        print(results_2)
        print(results_3)
        print(results_4)
        print("############# Fin ", i, " ième Frame(s) ##################")
        
        cv2.rectangle(frame, (facebox[0], facebox[1]),
                    (facebox[2], facebox[3]), (0, 255, 0), 5)
        label = results_2
        label_size, base_line = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)

        cv2.rectangle(frame, (facebox[0], facebox[1] + 10 - label_size[1]),
                        (facebox[0] + 90 + label_size[0],
                        facebox[1] - 75 + base_line),
                        (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, 'Name: '+ label, (facebox[0], facebox[1] - 55),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)
        
        cv2.putText(frame,'Head: '+ results_1, (facebox[0], facebox[1] - 35),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)
        
        cv2.putText(frame,'Emotion: '+ results_3, (facebox[0], facebox[1] - 15),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)
        
    cv2.rectangle(frame, (0, 0), (350, 40), (0, 0, 0), -1)
    cv2.putText(frame,'Activity: ' + results_4 , (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 255, 255), 2)
    
    cv2.imwrite("SAVE_PATH/img_multitask-{}.jpg".format(np.random.randint(0,100000)), frame)
    #cv2.imshow("Cheat demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()