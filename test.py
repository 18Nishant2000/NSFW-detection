import os
import tensorflow as tf
import cv2
import numpy as np

size = (100, 100)

path = './lmodel'
model1 = tf.keras.models.load_model(os.path.join(path, 'accuracy'))
model2 = tf.keras.models.load_model(os.path.join(path, 'val_accuracy'))

path = './tt'
p1 = []
p2 = []
print(os.listdir(path))
for i in os.listdir(path):
    img = cv2.imread(os.path.join(path, i))
    img = cv2.resize(img, size)
    img = np.array([img])
    img = img/255.0
    r1 = list(model1.predict(img)[0])
    r2 = list(model2.predict(img)[0])
    p1.append(r1.index(max(r1)))
    p2.append(r2.index(max(r2)))
    
print(p1)
print(p2)