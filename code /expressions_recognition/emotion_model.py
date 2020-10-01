# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2
from mtcnn.mtcnn import  MTCNN
from tensorflow.keras.preprocessing import image
import preprocessing as p

#load dadasets
with open("/home/kevin/spyder-workspace/EXAM_MONITORING/datasets/fer2013.csv") as f:
    data = f.readlines()

data = np.array(data)
print(data.size)

#0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
labels = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

num_classes = len(labels);
def model_networks(data, num_classes):
    # split data train and test sets
    x_train, y_train, x_test, y_test = [], [], [], []
    
    for i in range(1, data.size): 
      try:
          emotion, img, usage = data[i].split(",")
         
          val = img.split(" ")
          pixels = np.array(val, dtype='float32')
          pixels = pixels.reshape(48,48) # change shape from 2034 to 48*48
          emotion = tf.keras.utils.to_categorical(emotion, num_classes)
         
          if 'Training' in usage:
           y_train.append(emotion)
           x_train.append(pixels)
          elif ('PublicTest' in usage or 'PrivateTest' in usage):
           y_test.append(emotion)
           x_test.append(pixels)
      except:
         print("Error spliting data", end="")
    
    # lenght train and test
    print(len(x_train))
    print(len(x_test))
    
    x_train = np.array(x_train, 'float32')
    y_train = np.array(y_train, 'float32')
    x_test = np.array(x_test, 'float32')
    y_test = np.array(y_test, 'float32')
    
    x_train /= 255 #normalize inputs between [0, 1]
    x_test /= 255
    
    x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
    x_test = x_test.astype('float32')
    
    train_size = len(x_train)
    
    # @build model 
      
    epochs = 20
    batch_size = 64
    
    model = tf.keras.Sequential([
    #1st convolution layer
    tf.keras.layers.Conv2D(64, (5, 5), activation=tf.nn.relu, input_shape=(48,48,1)),
    tf.keras.layers.MaxPooling2D(pool_size=(5,5), strides=(2, 2)),
    #2nd convolution layer
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.AveragePooling2D(pool_size=(3,3), strides=(2, 2)),
    #3rd convolution layer
    tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.AveragePooling2D(pool_size=(3,3), strides=(2, 2)),

    tf.keras.layers.Flatten(),

    #fully connected neural networks
    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
    
    ])
    
    generator = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)
    train_generator = generator.flow(x_train, y_train, batch_size=batch_size)
     
    model.compile(loss='categorical_crossentropy'
    , optimizer=tf.keras.optimizers.Adam()
    , metrics=['accuracy']
    )
    
    model.load_weights("facial_expression_model_weights.h5")
    
    model.summary()
    
    model.fit(train_generator, steps_per_epoch=train_size//batch_size, epochs=epochs, validation_steps = (train_size*0.2)/batch_size)
    
    # evaluate modalel
    train_score = model.evaluate(x_train, y_train, verbose=0)
    print('Train loss:', train_score[0])
    print('Train accuracy:', 100*train_score[1])
     
    test_score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', test_score[0])
    print('Test accuracy:', 100*test_score[1])
    
    # save the model to disk
    #filename = "/home/kevin/spyder-workspace/EXAM_MONITORING/models_saved/facial_expression_recogintion_model.h5"
    #joblib.dump(model, filename)
    model.save("/home/kevin/spyder-workspace/EXAM_MONITORING/models_saved/facial_expression_recogintion_model.h5", save_format="h5")

#create and train model

model_networks(data, num_classes)

# Test from video streaming
"""
VIDEO_PATH = "/home/kevin/Downloads/test_exam.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

detector = MTCNN()
model = tf.keras.models.load_model("/home/kevin/spyder-workspace/EXAM_MONITORING/models_saved/facial_expression_recogintion_model.h5")

while cap.isOpened(): # True:
    ret, frame = cap.read()
    frame = p.frame_image(frame)
    # detect faces in the image
    faces = detector.detect_faces(frame)

    for face in faces:

        x, y, h, w = face['box']
       # gray_face = gray_image[h:w, x:y]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        detected_face = frame[int(y):int(y+h), int(x):int(x+w)] #crop detected face
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
        detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
        
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
         
        img_pixels /= 255
         
        predictions = model.predict(img_pixels)
         
        #find max indexed array
        max_index = np.argmax(predictions[0])
         
        emotion = labels[max_index]
         
        cv2.putText(frame, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        cv2.imshow('window_frame', frame)
        cv2.imwrite("/home/kevin/spyder-workspace/EXAM_MONITORING/expressions_recognition/outputs/img_emotion-{}.jpg".format(np.random.randint(0,1000)), frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

"""
