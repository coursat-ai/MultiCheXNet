import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import pydicom
import cv2
import keras
from keras.applications.vgg16 import preprocess_input


class Seg_gen(keras.utils.Sequence):
    'Generates data from a Dataframe'
    def __init__(self, df, preprocess_fct, batch_size=32, dim=(1024, 1024), shuffle=True):
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
        
        X = np.empty((self.batch_size, *self.dim))
        Y = np.empty((self.batch_size, *self.dim))
        
        # Generate data
        for i, ID in enumerate(index):
            # Read the image
            img  = pydicom.dcmread(self.df['full_path'][ID]).pixel_array
            mask = rle2mask(self.df[' EncodedPixels'][ID], *img.shape).T/255           
                                                                                
            img  = np.asarray(cv2.resize(img, self.dim))
            mask = np.asarray(cv2.resize(mask, self.dim))            
            
            X[i,] = np.asarray(img)#self.preprocess_fct(np.asarray(img))
            Y[i,] = np.asarray(mask)
        
        return X, Y 


x,y = next(iter(Seg_gen(rle_csv, preprocess_input, batch_size = 32, dim=(256,256))))
