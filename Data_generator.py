import numpy as np
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from glob import glob

path = '/kaggle/input/data/'

bbox_csv = 'BBox_List_2017.csv'
data_csv = 'Data_Entry_2017.csv'

imgs_folders = ['images_001', 'images_002', 'images_003',
       'images_004', 'images_005', 'images_006', 'images_007',
       'images_008', 'images_009', 'images_010', 'images_011',
       'images_012']

Categories = [ 'No Finding','Atelectasis', 'Cardiomegaly', 'Consolidation',
        'Edema', 'Effusion', 'Fibrosis', 'Infiltration', 'Mass',
        'Pneumothorax', 'Emphysema', 'Pneumonia', 'Pleural_Thickening',
        'Nodule', 'Hernia']

data_df = pd.read_csv(path + data_csv)
box_df  = pd.read_csv(path + bbox_csv)



my_glob = glob('/kaggle/input/data/images*/images/*.png')

full_img_paths = {os.path.basename(x): x for x in my_glob}
dataset_path = data_df['Image Index'].map(full_img_paths.get)


def Adjust_data(data, box):
    data = data[['Image Index', 'Finding Labels']]
    box = Adjust_box(box)
    new_data = pd.merge(left=data, right=box, left_on ='Image Index', right_on ='Image Index', how = 'left')
    new_data = new_data.rename(columns={'Finding Labels': 'All Labels', 'Finding Label' : "Det Label"})
    new_data = new_data.fillna({'Det Label': 'No Finding', 'Bbox':0})
    new_data['full_path'] = dataset_path
    
    return new_data ,box  

def Adjust_box(box_df):
    box_df['Finding Label'] = box_df['Finding Label'].str.replace('Infiltrate','Infiltration')
    
    #Sorting by Image index (not needed but helps in debugging)
    box_df['sort'] = box_df['Image Index'].str.extract('(\d+)', expand=False).astype(int)
    box_df.sort_values('sort',inplace=True, ascending=True)
    box_df = box_df.reset_index(drop=True)
    box_df = box_df.drop(['sort','Unnamed: 6','Unnamed: 7','Unnamed: 8'], axis=1)
    
    #Getting all values of x, y ,w ,h
    x= box_df.groupby('Image Index')['Bbox [x'].apply(np.array).reset_index()['Bbox [x'].values
    y= box_df.groupby('Image Index')['y'].apply(np.array).reset_index()['y'].values
    w= box_df.groupby('Image Index')['w'].apply(np.array).reset_index()['w'].values
    h= box_df.groupby('Image Index')['h]'].apply(np.array).reset_index()['h]'].values
    box_df= box_df.groupby('Image Index')['Finding Label'].apply('|'.join).reset_index()
    
    #Arranging the bounding boxes values into arrays
    bbox1 = []
    for i in range(len(x)):
        bbox2 = []
        for a,b,c,d in zip(x[i], y[i], w[i], h[i]):
            bbox2.append(np.array([a, b, c, d]))
        bbox1.append(bbox2)
        
    #Concatenating the bounding boxes into 1 line of string format
    mbbs = [' '.join(str(x) for p in o for x in p) for o in bbox1]
    
    box_df["Bbox"] = mbbs
    return box_df

def dfcat2dfid(df, Categories): # change dataframe of category names into category numbers
    cat2id = {i:j for j,i in enumerate(Categories)}
    id2cat = {i:j for i,j in enumerate(Categories)}

    All_Cat = df['All Labels'].values.astype(str)
    All_cat_list = [i.split('|') for i in All_Cat]

    Det = df['Det Label'].values.astype(str)
    Det_list = [i.split('|') for i in Det]


    mcs_All_Labels = np.array([np.array([cat2id[p] for p in o]) for o in All_cat_list])
    mcs_Det_Labels = np.array([np.array([cat2id[p] for p in o]) for o in Det_list])

    df['Class_All'] = mcs_All_Labels
    df['Class_Det'] = mcs_Det_Labels
    
    return df



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
            img = cv2.imread(df['full_path'][ID])
            
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


train_gen = Data_gen(df, preprocess_input, batch_size=10, shuffle=True)

x, y = next(iter(train_gen))