# -*- coding: utf-8 -*-
from PIL import Image
from numpy import asarray, expand_dims
import inception_resnet_v1 as inception

def input_setup(file):
    
    img = Image.open(file)
    img = img.convert('RGB')
    img = asarray(img)
 
    return img

def facenet_model():
    model = inception.InceptionResNetV1()
    model.load_weights('facenet_weights.h5')
 
    model.summary()
    
    return model
    
    
def image_resize(img, img_size=(160, 160)):
    img = img.resize(img_size)
    img = asarray(img)
    
    return img

def face_embedding(model, face):
    
    face = face.astype('float32')
    #normalize pixels values
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    face = expand_dims(face, axis=0)
    embedding = model.predict(face)
    
    return embedding[0]