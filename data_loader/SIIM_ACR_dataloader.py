import os
import numpy as np
import pandas as pd
from glob import glob
import pydicom
import tensorflow
from skimage.transform import resize
import random
import cv2
from skimage import exposure
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,OneOf,
    ToFloat, GridDistortion, ElasticTransform, OpticalDistortion, 
    RandomSizedCrop,Rotate
)

h,w = 256,256
AUGMENTATIONS_TRAIN = Compose([
    HorizontalFlip(p=0.5),
    OneOf([
        
        RandomGamma(),
        RandomBrightness(),
         ], p=0.3),
    OneOf([
        ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        GridDistortion(),
        OpticalDistortion(distort_limit=2, shift_limit=0.5),
        ], p=0.3),
    Rotate(limit=15, p=0.2),
    RandomSizedCrop(min_max_height=(176, 256), height=h, width=w,p=0.25)],p=1)


def add_full_path(df, train_path):
    my_glob = glob(train_path + '/*/*/*.dcm')

    full_img_paths = {os.path.basename(x).split('.dcm')[0]: x for x in my_glob}
    dataset_path = df['ImageId'].map(full_img_paths.get)

    df['full_path'] = dataset_path

    return df


def rle2mask(rle, width, height):
    if rle == ' -1':
        rle = '0 0'

    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position + lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)

def masks_as_image(rle_list, shape):
    # Take the individual masks and create a single mask array
    all_masks = np.zeros(shape, dtype=np.uint8)
    for mask in rle_list:
        if isinstance(mask, str) and mask != '-1':
            all_masks |= rle2mask(mask, shape[0], shape[1]).T.astype(bool)
    return all_masks


class Seg_gen(tensorflow.keras.utils.Sequence):
    'Generates data from a Dataframe'

    def __init__(self, df_path,patient_ids, train_path, preprocess_fct=None, batch_size=32, dim=(256, 256), shuffle=True , n_channels=3,augmentation=None,normalize=False,hist_eq=False,batch_positive_portion=None,split_type='train'):
        'Initialization'
        
        rle_csv = pd.read_csv(df_path)
    
        self.df = add_full_path(rle_csv, train_path)
        
        self.patient_ids = patient_ids
        
        self.preprocess_fct = preprocess_fct
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.normalize = normalize
        self.augmentation= augmentation
        self.hist_eq = hist_eq
        
        self.split_type=split_type

        self.n = len(self.patient_ids)
        self.nb_iteration = int(np.floor(self.n / self.batch_size))
        
        self.n_channels = n_channels
        self.batch_positive_portion= batch_positive_portion
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.nb_iteration

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch        
        
        # Generate data
        X, y = self.__data_generation(index)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        random.shuffle(self.patient_ids)

    def __data_generation(self, index):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        
        if self.batch_positive_portion==None or self.split_type!='train':
            indexes = range(index*self.batch_size, min((index*self.batch_size)+self.batch_size ,len(self.patient_ids) ))
            patient_ids= self.patient_ids[indexes]
        else: 
            filtered_df = self.df[self.df["ImageId"].isin(self.patient_ids)]
            filtered_df_positive =  filtered_df[filtered_df[" EncodedPixels"]!=' -1']
            filtered_df_negative =  filtered_df[filtered_df[" EncodedPixels"]==' -1']
            
            positive_ids= filtered_df_positive['ImageId'].tolist()
            negative_ids= filtered_df_negative['ImageId'].tolist()
            
            random.shuffle(positive_ids)
            random.shuffle(negative_ids)
            
            num_positive = int((self.batch_size)*self.batch_positive_portion)
            num_negative = self.batch_size - num_positive
            
            patient_ids= positive_ids[:num_positive]
            patient_ids+= negative_ids[:num_negative]
            
            random.shuffle(patient_ids)
            
            
            

        X = []#np.empty((self.batch_size, self.dim[0],self.dim[1],self.n_channels))
        Y = []#np.empty((self.batch_size,  self.dim[0],self.dim[1]))
        
        
        # Generate data
        for i, ID in enumerate(patient_ids):
            # Read the image
            filtered_dataframe = self.df[self.df["ImageId"]==ID]
            
            img = pydicom.dcmread(filtered_dataframe['full_path'].iloc[0]).pixel_array
            if filtered_dataframe[" EncodedPixels"].iloc[0]==' -1':
                mask= np.zeros(self.dim)
            else:
                mask = masks_as_image(filtered_dataframe[' EncodedPixels'], (1024,1024))
            
            
            if self.n_channels == 3:
                img = cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
            
            img = np.asarray(cv2.resize(img, self.dim))

            if self.preprocess_fct !=None:
                img= self.preprocess_fct(img)
            
            mask = resize(mask,
                          self.dim,
                          mode='edge',
                          anti_aliasing=False,
                          anti_aliasing_sigma=None,
                          preserve_range=True,
                          order=0)
            
            
            if self.augmentation=='train':
                aug= AUGMENTATIONS_TRAIN(image=img,mask=mask)
                img=aug['image']
                mask=aug['mask']
            
            
            if self.hist_eq:
                img= exposure.equalize_adapthist(img)
            
            
            if self.normalize and img.max()>1:
                img=np.array(img,np.float32)/255
            
            
            
            X.append(np.asarray(img))  # self.preprocess_fct(np.asarray(img))
            Y.append(np.asarray(mask, dtype=np.int8))
        
        return np.array(X), np.expand_dims(np.array(Y),axis=3)
    
    
def get_train_validation_generator(csv_path,img_path ,batch_size=8, dim=(256,256), n_channels=3, shuffle=True ,preprocess = None , only_positive=True, validation_split=0.2,augmentation=False,normalize=False,hist_eq =False,batch_positive_portion=None ):

  df = pd.read_csv(csv_path)
  if only_positive:
    df = df[df[" EncodedPixels"]!=' -1']

  random.seed(42)
  patient_ids = df["ImageId"].unique()
  random.shuffle(patient_ids)

  patient_ids_train = patient_ids[int(len(patient_ids)*validation_split ):]
  patient_ids_validation = patient_ids[: int(len(patient_ids)*validation_split)]

  if augmentation == True:
        augmentation='train'
  train_gen = Seg_gen(csv_path,patient_ids_train , img_path ,batch_size=batch_size, dim=dim, n_channels=n_channels,
                         shuffle=shuffle, preprocess_fct = preprocess,augmentation=augmentation, normalize=normalize,hist_eq=hist_eq,batch_positive_portion=batch_positive_portion,split_type='train')

  if augmentation == 'train':
        augmentation='validation'
        
  validation_gen = Seg_gen(csv_path, patient_ids_validation, img_path, batch_size=batch_size, dim=dim, n_channels=n_channels,
                       shuffle=shuffle,  preprocess_fct=preprocess,augmentation=augmentation, normalize=normalize,hist_eq=hist_eq,batch_positive_portion=batch_positive_portion,split_type='validation')


  return train_gen, validation_gen
