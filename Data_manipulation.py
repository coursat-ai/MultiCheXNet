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


df, Xbox = Adjust_data(data_df, box_df)

df = dfcat2dfid(df, Categories)

