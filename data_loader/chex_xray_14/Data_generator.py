import numpy as np
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from glob import glob

class Data_gen(keras.utils.Sequence):
    'Generates data from a Dataframe'
    def __init__(self, df, preprocess_fct, batch_size=32, dim=(256,256), shuffle=True):
        'Initialization'
        self.preprocess_fct = preprocess_fct
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.df = df
        self.n = len(df)            
        self.nb_iteration = int(np.floor(self.n  / self.batch_size))
        
        self.on_epoch_end()
                    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.nb_iteration

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y
   
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        nb_label_max = 7
        X = np.empty((self.batch_size, *self.dim, 3))
        Y = []
        
        # Generate data
        for i, ID in enumerate(index):
            # Read the image
            img = cv2.imread(self.df['full_path'][ID])
            
            # extract the number of label
            c = self.df['Class_Det'][ID]
            nb_label = len(c)            
            
            # Class in a form of a one hot encoding
            y = np.zeros((nb_label_max,1+4));
            y[:nb_label,0] = c
            
            # reshape the bounding box and resize
            if self.df['Bbox'][ID] != 0:
                bbox = np.asmatrix(self.df['Bbox'][ID]).reshape((nb_label,4))
            else: bbox = np.zeros((nb_label,4))
            
            bbox_rescaled = np.copy(bbox)    
            bbox_rescaled = bbox_rescaled.astype(float)
            width, height = img.shape[0:2]
            RatioX = width/self.dim[0]
            RatioY = height/self.dim[1]

            bbox_rescaled[:,0] = bbox_rescaled[:,0]/RatioY/self.dim[1]
            bbox_rescaled[:,1] = bbox_rescaled[:,1]/RatioX/self.dim[0]
            bbox_rescaled[:,2] = bbox_rescaled[:,2]/RatioY/self.dim[1]
            bbox_rescaled[:,3] = bbox_rescaled[:,3]/RatioX/self.dim[0]        
        
            # save the bb coordinates
            y[:nb_label,1:5] = bbox_rescaled
            
       
            # reshape to a vector
            y=np.reshape(y,nb_label_max*5)
                                                                    
            img = np.asarray(cv2.resize(img, self.dim))
            X[i,] = np.asarray(img)#self.preprocess_fct(np.asarray(img))
            
            Y.append(np.asarray(y))

        Y = np.asarray(Y)
        
        return X, Y 


#train_gen = Data_gen(df, preprocess_input, batch_size=10, shuffle=True)

#x, y = next(iter(train_gen))
