#!/usr/bin/env python3
import os
import cv2
import dlib
import numpy as np

# helper modules
from drawFace import draw
import reference_world as world
from imutils import face_utils
from tensorflow.keras.models import load_model
import preprocessing as p
#import main_pre_processing as preprocess

PREDICTOR_PATH = os.path.join("models", "shape_predictor_68_face_landmarks.dat")


focal = 60
image = "data/images/01.jpg"
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

def set_color(pose):
    
    if pose == "Right" or "Left":
        color = (0, 0, 255)
    elif pose == "Normal":
        color = (0, 255, 0)
    
    return color

# Load emotions model for also predict emotion
FER_model = load_model("/home/kevin/spyder-workspace/EXAM_MONITORING/models_saved/facial_expression_recogintion_model")

def main(source=0):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    cap = cv2.VideoCapture("/home/kevin/Downloads/test_exam.mp4")

    while True:
        head = "Face Not Found"
        ret, frame = cap.read()
        if not ret:
            print("[ERROR - System]Cannot read from source")
            break
        frame = p.frame_image(frame)
        faces = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 0)
       
        face3Dmodel = world.ref3DModel()

        for face in faces:
            shape = predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), face)

            draw(frame, shape)

            refImgPts = world.ref2dImagePoints(shape)

            height, width, channels = frame.shape
            focalLength = focal * width
            cameraMatrix = world.cameraMatrix(focalLength, (height / 2, width / 2))

            mdists = np.zeros((4, 1), dtype=np.float64)

            # calculate rotation and translation vector using solvePnP
            success, rotationVector, translationVector = cv2.solvePnP(
                face3Dmodel, refImgPts, cameraMatrix, mdists)

            noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
            noseEndPoint2D, jacobian = cv2.projectPoints(
                noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists)

            #  draw nose line
            p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
            p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
            cv2.line(frame, p1, p2, (110, 220, 0),
                     thickness=2, lineType=cv2.LINE_AA)

            # calculating euler angles
            rmat, jac = cv2.Rodrigues(rotationVector)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            head = "head pose: "
            oriented = "Normal"
            if angles[1] < -15:
                oriented = "Left"
            elif angles[1] > 15:
                oriented = "Right"
            
            head += oriented
            (x, y, w, h) = face_utils.rect_to_bb(face)
          
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), set_color(oriented), 2)            
           
            cv2.putText(frame, head, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

      
            cv2.imshow("Head Pose", frame)
            cv2.imwrite("/home/kevin/spyder-workspace/EXAM_MONITORING/head_pose_recognition/outputs/img_head-pose-{}.jpg".format(np.random.randint(0,1000)), frame)
        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("head_pose_poc.webm")
