# -*- coding: utf-8 -*-
from mtcnn.mtcnn import MTCNN
from os import listdir
from os.path import isdir
from PIL import Image
from numpy import asarray
import pre_processing_facenet as facenet

def mtcnn_model(frame):
    detector = MTCNN()
    boundings_results = detector.detect_faces(frame)
    
    return boundings_results

def extract_faces(file):
    
    img = facenet.input_setup(file)
    faces_bounding = mtcnn_model(img)
    # bounding extracted
    x1, y1 , width, height = faces_bounding[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # face extracted
    face = img[y1:y2, x1:x2]
    # resize image
    face = Image.fromarray(face)
    face = facenet.image_resize(face)
    
    return face

def load_class_to_extract_faces(class_directory):
    
    faces = list()
    for filename in listdir(class_directory):
        # get path
        path_file = class_directory + filename
        #get face
        face = extract_faces(path_file)
        # update list
        faces.append(face)
    
    return faces
    
def load_datasets(dataset_directory):
    
    list_of_list_faces, labels_of_list_faces = list(), list()
    for folder in listdir(dataset_directory):
        #get path
        path_folder = dataset_directory + folder + '/'
        #load faces in the path_folder
        if not isdir(path_folder):
            continue
        faces = load_class_to_extract_faces(path_folder)
        # get faces labels
        labels = [folder for _ in range(len(faces))]
        #debug
        print(">loaded {} examples for class: {}" .format(len(faces), folder))
        # update list
        list_of_list_faces.extend(faces)
        labels_of_list_faces.extend(labels)
        
    return asarray(list_of_list_faces), asarray(labels_of_list_faces)
        

    
    