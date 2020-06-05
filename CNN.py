# -*- coding: utf-8 -*-
"""
Created on Sun May 31 15:10:22 2020

@author: rksha
"""


import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense 
from keras.preprocessing.image import ImageDataGenerator
train = ImageDataGenerator(rescale = 1./225,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
train_set = train.flow_from_directory('dataset/training_set',target_size = (64,64),batch_size = 32,class_mode = 'binary')
test = ImageDataGenerator(rescale = 1./225)
test_set = test.flow_from_directory('dataset/test_set',target_size = (64,64),batch_size = 32,class_mode = 'binary')
clf = Sequential()
clf.add(Conv2D(32,(3,3),activation = 'relu',input_shape = (64,64,3)))
clf.add(MaxPooling2D(pool_size = 2 ,strides = 2))
clf.add(Conv2D(32,(3,3),activation = 'relu'))
clf.add(MaxPooling2D(pool_size = 2 ,strides = 2))
clf.add(Flatten())
clf.add(Dense(units = 128,activation = 'relu'))
clf.add(Dense(units = 1 , activation = 'sigmoid'))
clf.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
clf.fit_generator(train_set,samples_per_epoch = 8000,nb_epoch = 25,validation_data = test_set,nb_val_samples = 2000)
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',target_size = (64,64)) 
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = clf.predict(test_image)
train_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
