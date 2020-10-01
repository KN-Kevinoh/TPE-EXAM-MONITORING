#!/usr/bin/env python3

import os
import cv2
import sys
import dlib
import numpy as np

from drawFace import draw
import reference_world as world
from imutils import face_utils
from tensorflow.keras.models import load_model
import main_pre_processing as preprocess

PREDICTOR_PATH = os.path.join("models", "shape_predictor_68_face_landmarks.dat")

if not os.path.isfile(PREDICTOR_PATH):
    print("[ERROR] No directory")
    sys.exit()

focal = 60
image = "data/images/etudiants.jpg"
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

#set color
def set_color(pose):
    
    if pose == "Right" or "Left":
        color = (0, 0, 255)
    elif pose == "Normal":
        color = (0, 255, 0)
    
    return color

# Load emotions model for also predict emotion
FER_model = load_model("/home/kevin/spyder-workspace/EXAM_MONITORING/models_saved/facial_expression_recogintion_model")

def main(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    while True:
        im = cv2.imread(image)

        faces = detector(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), 0)
        gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        face3Dmodel = world.ref3DModel()

        for i,face in enumerate(faces):
            shape = predictor(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), face)

            draw(im, shape)

            refImgPts = world.ref2dImagePoints(shape)

            height, width, channel = im.shape
            focalLength = focal * width
            cameraMatrix = world.cameraMatrix(focalLength, (height / 2, width / 2))

            mdists = np.zeros((4, 1), dtype=np.float64)

            # calculate rotation and translation vector using solvePnP
            success, rotationVector, translationVector = cv2.solvePnP(
                face3Dmodel, refImgPts, cameraMatrix, mdists)

            noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
            noseEndPoint2D, jacobian = cv2.projectPoints(
                noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists)

            # draw nose line 
            p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
            p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
            cv2.line(im, p1, p2, (110, 220, 0),
                     thickness=2, lineType=cv2.LINE_AA)

            # calculating angle
            rmat, jac = cv2.Rodrigues(rotationVector)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            print(angles)
            print("-------------------")
            head = "head pose: "
            oriented = "Normal"
            if angles[1] < -15:
                oriented = "Left"
            elif angles[1] > 15:
                oriented = "Right"
            
            head += oriented
            (x, y, w, h) = face_utils.rect_to_bb(face)
            
            # FER checked

            gray_face = gray_image[w:h, x:y]
            
            #gray_face = cv2.resize(gray_face, (48, 48))
          
            #cv2.imwrite("face.png", gray_face)
            #gray_face = preprocess.preprocess_input(gray_face)
            gray_face = preprocess.img_emotion_resize(gray_face, i)
            #gray_face = np.expand_dims(gray_face, -1)
                    
            
            #f = preprocess.img_emotion_resize(face,i)
            pred_emotion = FER_model.predict(gray_face) 
            print("kevin: ", pred_emotion)
            #name = preprocess(pred_emotion[0])
            cv2.rectangle(im, (x, y), (x + w, y + h), set_color(oriented), 2)            
           
            cv2.putText(im, head, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            #cv2.putText(im, gaze, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
            cv2.imshow("Head Pose", im)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            cv2.imwrite("joye-{}.jpg".format(head), im)
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(image)
