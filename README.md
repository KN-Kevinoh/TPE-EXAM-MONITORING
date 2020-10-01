TPE - Exam monitoring


In this work we make four models:

-Face recognition model based on faceNet model
-Facial expression recognition model based on CNN
-Cheating Activities recognition model based respectively on resnet50, vgg16, vgg19, inception and xception models
-Head pose model based on simple CNN model 

Description: 


- Our FR use faceNet(InceptionResNetV1) model with pre-trained weights to benefit transfer learning and SVM. We use lfw dataset.

- Our FER is use seven classes (sad, happy, surprise, fear,  disgust, angry,      neutral) to classify faces expressions.
We use FER2013 dataset and add another dataset images to have a dataset of 
49569 images.
   
-Our cheating activities recognition use five classes (No cheat, exchange paper, use cheat sheet, looking at friend and talking friend ) to classify eventually cheats actions.
We made our own dataset of 294 images.
We inspired on pre-trained models in the popular ImageNet dataset to classify images into 1000 classes,  include in keras such as vgg16, vgg19, resnet50,inception and xception in tensorflow, to benefit transfer learning.
We have achieve a good performance with resnet50, vgg16, inception and particularly with xception. 

- Our head pose models reuse https://github.com/arnaldog12/Deep-Learning/blob/master/problems/Regressor-Face%20Pose/Keras.ipynb approach with a little beat change on CNN model base on classes Roll, Pitch and Yaw Euler angles.

Project folders:

- in directory models you can check our saved models and share link for heavy models .

- in directory datasets, here is the datasets we are used.

- in colab directory some of our project files use Jupiter notebook.

- in code directory our FR in python.


NB: In this project we built and tested models for helping monitor  exam, the application come soon.
