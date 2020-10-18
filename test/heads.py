from os import listdir
from numpy import asarray
from keras.models import load_model
import numpy as np
import cv2 
import tensorflow as tf
import dlib
from sklearn.preprocessing import StandardScaler
import math


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


def detect_face_points(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_rect = detector(image, 1)
    faces_points = []
    face_pos = []
    for face in face_rect:
        
        x = face.left()
        y = face.top()
        x1 = face.right()
        y1 = face.bottom()
        cv2.rectangle(image, (x, y), (x1, y1), (0, 0, 255), 2)
        face_pos.append(np.array([x,y]))
        
        #if len(face_rect) != 1: return []
        
        dlib_points = predictor(image, face)
        face_points = []
        for i in range(68):
            a, b = dlib_points.part(i).x, dlib_points.part(i).y
            face_points.append(np.array([a, b]))
        
        faces_points.append(face_points)
    
    return  image, face_pos, faces_points

        
def compute_features(face_points):
    assert (len(face_points) == 68), "len(face_points) must be 68"
    
    face_points = np.array(face_points)
    features = []
    for i in range(68):
        for j in range(i+1, 68):
            features.append(np.linalg.norm(face_points[i]-face_points[j]))
            
    return np.array(features).reshape(1, -1)


  
  
VIDEO_PATH = "/home/kevin/Videos/Webcam/2020-10-17-110959.webm"
cap = cv2.VideoCapture(VIDEO_PATH)

model = load_model("models/head_pose_model.h5")
# head pose train
data= np.load("datasets/head-train.npz")
data_train = data['arr_0']
std = StandardScaler()

std.fit(data_train)

def label_head(cible):
   
    if cible < 0:
        return "left"
    else:
        return "right"
    

while cap.isOpened(): # True:
    ret, frame = cap.read()
    # detect faces in the image
    
    
    #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    frame = image_processing(frame)
    
    if ret == False:
        break
   
    #im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = frame
    frame, face_pos, faces_points = detect_face_points(im)
    
    for face_points, pos in zip(faces_points, face_pos):
        pos = np.array(pos, dtype=np.int)
        #for a, b in face_points:
            #cv2.circle(im, (a, b), 1, (0, 255, 0), -1)
            
        features = compute_features(face_points)
       
        features = std.transform(features)
        
        y_pred = model.predict(features)

        roll_pred, pitch_pred, yaw_pred = y_pred[0]
        print("predict : ", y_pred[0] )
        # Get the absolute value for each number
        absValues = [math.fabs(number) for number in y_pred[0]]
        absValues = np.array(absValues)
        print("absValues :", absValues)
        max_val = max(absValues)
        print("max_val :", max_val)
        label = "Normal"
        if max_val == math.fabs(roll_pred): 
            label = label_head(roll_pred)
        
        if max_val == math.fabs(yaw_pred): 
            label = label_head(yaw_pred)
            
        print(label)
        cv2.putText(frame, label, (pos[0], pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        cv2.imwrite("SAVE_PATH/img_heads-{}.jpg".format(np.random.randint(0,1000)), frame)
        #cv2.imshow("HP demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
