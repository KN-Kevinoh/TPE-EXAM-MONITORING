# -*- coding: utf-8 -*-
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt



def img_emotion_resize(face, i):
    #f = np.float32(cv2.UMat(f))
    #x  = image.load_img("face.png", grayscale=True, target_size=(48, 48))
    x = face.reshape([48,48]);
    x = np.expand_dims(face, axis = 0)
    x = np.float32(x)
    
    x /= 255
    plt.gray()
    plt.imshow(x)
    plt.savefig(str(i) + "_face" + "png")
    return np.asarray(x)

def emotion_analysis(emotion, i):
    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(emotions))
 
    plt.bar(y_pos, emotion, align='center', alpha=0.5)
    plt.xticks(y_pos, emotions)
    plt.ylabel('percentage')
    plt.title('emotion')
    plt.savefig(str(i) + "_emotion" + "png")
 
    return emotions[emotion]
    
def preprocess_input(x):
    x = x.astype('float32')
    x = x / 255.0
   
    return x

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)
