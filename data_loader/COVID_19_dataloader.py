from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator
import math
import os
from glob import glob
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
    RandomSizedCrop(min_max_height=(190, 256), height=h, width=w,p=0.25)],p=1)




class dicom_data_generator(Sequence):
  def __init__(self, df,dataset_path,dim=(256,256),batch_size=16,randomize=True,split="training", normalize=False,hist_eq=False, data_aug=False):
    self.dataset_path = dataset_path
    self.batch_size = batch_size
    self.dim=dim
    self.randomize=randomize

    self.normalize = normalize
    self.data_aug = data_aug 
    self.hist_eq= hist_eq


    if split == "training":
      self.split=1
    elif split == "validation":
      self.split=2

    self.df = df[df["split"]==self.split]

    self.num_samples =(df["split"]==self.split).sum()
    self.iterations_per_batch = math.ceil(self.num_samples/self.batch_size)

  def __len__(self):
    return self.iterations_per_batch

  def __getitem__(self,index):
    X=[]
    Y=[]
    
    if self.randomize==True:
      indicies = np.random.randint(self.num_samples,size=self.batch_size)
    else:
      indicies= list(range(index*self.batch_size , min(index*self.batch_size+self.batch_size ,self.num_samples) ))

    for index in indicies:
      
      filename = self.df.iloc[index]["filename"]

      file_path = os.path.join(self.dataset_path,filename)
      
      
      img=cv2.imread(file_path , cv2.IMREAD_GRAYSCALE)
      img= cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
      img = np.asarray(cv2.resize(img, self.dim))

      if self.data_aug == True and self.split==1:
        aug= AUGMENTATIONS_TRAIN(image=img)
        img=aug['image']

      if self.hist_eq:
        img= exposure.equalize_adapthist(img)
            
            
      if self.normalize and img.max()>1:
          img=img/255



      X.append(img)
      Y.append(self.df.iloc[index]["class"])  
      

    X = np.array(X)
    return X ,np.expand_dims(np.array(Y),axis=-1)
    
    
def get_train_validation_generator(csv_path,img_path,batch_size=8, dim=(256,256),shuffle=True , validation_split=0.2,augmentation=False,normalize=False,hist_eq=False):
    csv_path = "covid-chestxray-dataset/metadata.csv"
    img_path= "covid-chestxray-dataset/images/"
    df= pd.read_csv(csv_path)

    df =df[df['filename'].str.strip().str[-2:]!='gz']
    df["class"] = df["finding"].str.contains('COVID-19').astype(int)

    np.random.seed(42)
    prob = np.random.rand(df.shape[0])
    df["split"] = 0
    df["split"] = df["split"]+ ((prob>=validation_split)*1)
    df["split"] = df["split"]+((prob<validation_split)*2)
                               
    train_gen = dicom_data_generator(df,images_path ,
                                     batch_size=batch_size,
                                     split="training" ,
                                     dim=dim,
                                     normalize=normalize ,
                                     randomize=shuffle,
                                     hist_eq=hist_eq,
                                     data_aug=augmentation)
    
    valid_gen = dicom_data_generator(df,images_path ,
                                     batch_size=batch_size,
                                     split="validation",
                                     dim=dim,
                                     normalize=normalize,
                                     hist_eq=hist_eq,
                                     randomize=shuffle)

    return train_gen,valid_gen   
