# -*- coding: utf-8 -*-

from mtcnn_detector import load_datasets
from pre_processing_facenet import face_embedding, facenet_model
from numpy import savez_compressed, asarray
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import  confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools




#load facenet model
model = facenet_model()

# load train set
print('Load train set:')
trainX, trainY = load_datasets("/opt/Documents/I.F.I-Vietnam/COURS_IFI/TPE/My work/datasets/FR/train/")
print(trainX.shape, trainY.shape)

# load test set
print ('Load test set:')
testX, testY = load_datasets("/opt/Documents/I.F.I-Vietnam/COURS_IFI/TPE/My work/datasets/FR/test/")
print(testX.shape, testY.shape)

"""
    Training stage
    first get embedding for train and test set
"""
train_embedding, test_embedding = list(), list()
for train_face in trainX:
    train_embedding.append(face_embedding(model, train_face))
train_embedding = asarray(train_embedding)
print(train_embedding.shape)

for test_face in testX:
    test_embedding.append(face_embedding(model, test_face))
test_embedding = asarray(test_embedding)
print(test_embedding.shape)

# Save faces embeddings
savez_compressed("/opt/Documents/I.F.I-Vietnam/COURS_IFI/TPE/My work/datasets/FR/lfw-faces-embeddings.npz", train_embedding, trainY, test_embedding, testY)

"""
    Recognition stage by using SVM classifier
"""

# Normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(train_embedding)
testX = in_encoder.transform(test_embedding)

# Labels encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainY)
trainY = out_encoder.transform(trainY)
testY = out_encoder.transform(testY)

print("------------------------------------------")
# SVM classifier fitting
print("SVM classifier modle: ")
print("############################################")
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainY)

# Scores
print("Accuracy of SVM Classifier on training data set: {:.2f}" .format(model.score(trainX, trainY)))
print("Accuracy of SVM Classifier on testing data set: {:.2f}" .format(model.score(testX, testY)))

#confusion matrix
pred = model.predict(testX)
#print(confusion_matrix(testY,pred)) # matrix to describe a performance of classification model
#print(classification_report(testY,pred))


print("------------------------------------------")
# KNN classifier fitting
print("KNN classifier model")
print("############################################")
knn = KNeighborsClassifier()
knn.fit(trainX, trainY)

# Scores
print("Accuracy of KNN Classifier on training data set: {:.2f}" .format(knn.score(trainX, trainY)))
print("Accuracy of KNN Classifier on testing data set: {:.2f}" .format(knn.score(testX, testY)))

#confusion matrix
pred = knn.predict(testX)
#print(confusion_matrix(testY,pred)) # matrix to describe a performance of classification model
#print(classification_report(testY,pred))

    
print("------------------------------------------")
# decision tree classifier fitting
print("decision tree classifier model")
print("############################################")
dtc = DecisionTreeClassifier().fit(trainX, trainY)

# Scores
print("Accuracy of DTC Classifier on training data set: {:.2f}" .format(dtc.score(trainX, trainY)))
print("Accuracy of DTC Classifier on testing data set: {:.2f}" .format(dtc.score(testX, testY)))

#confusion matrix
pred = dtc.predict(testX)
#print(confusion_matrix(testY,pred)) # matrix to describe a performance of classification model
#print(classification_report(testY,pred))



print("------------------------------------------")
# Naive bayes classifier
print("Naive Bayes classifier model")
print("############################################")
bnc = GaussianNB().fit(trainX, trainY)

# Scores
print("Accuracy of BNC Classifier on training data set: {:.2f}" .format(bnc.score(trainX, trainY)))
print("Accuracy of BNC Classifier on testing data set: {:.2f}" .format(bnc.score(testX, testY)))

#confusion matrix
pred = bnc.predict(testX)
#print(confusion_matrix(testY,pred)) # matrix to describe a performance of classification model
#print(classification_report(testY,pred))
