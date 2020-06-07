import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import pydicom
import cv2
import keras
from keras.applications.vgg16 import preprocess_input


train_path = '/kaggle/input/siim-dicom-images/siim-original/dicom-images-train'
rle_path = '/kaggle/input/siim-dicom-images/train-rle.csv'

train_filenames = os.listdir(train_path)
rle_csv = pd.read_csv(rle_path)


def process_labels(df): #Converts labels to binary
    
    labels = df[" EncodedPixels"].to_list()
    a = []
    for i in labels:
      if i == ' -1':
        a.append(0)
      else: a.append(1)
    df["Class"] = np.array(a, dtype='uint8')
    return df

def add_full_path(df, train_path):
    my_glob = glob(train_path + '/*/*/*.dcm')

    full_img_paths = {os.path.basename(x).split('.dcm')[0]: x for x in my_glob}
    dataset_path = df['ImageId'].map(full_img_paths.get)
    
    df['full_path'] = dataset_path
    
    return df


def mask2rle(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;

    return " ".join(rle)

def rle2mask(rle, width, height):
    if rle == ' -1':
        rle ='0 0'
    
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)


#rle_csv = process_labels(rle_csv)
#rle_csv = add_full_path(rle_csv, train_path)
