import os
import pydicom
from skimage import exposure
import numpy as np
import tensorflow
import random
import pandas as pd
import cv2

def bbToYoloFormat(bb):
    """
    converts (left, top, right, bottom) to
    (center_x, center_y, center_w, center_h)
    """
    x1, y1, x2, y2 = np.split(bb, 4, axis=1) 
    w = x2 - x1
    h = y2 - y1
    c_x = x1 + w / 2
    c_y = y1 + h / 2
    return np.concatenate([c_x, c_y, w, h], axis=-1)
def findBestPrior(bb, priors):
    """
    Given bounding boxes in yolo format and anchor priors
    compute the best anchor prior for each bounding box
    """
    w1, h1 = bb[:, 2], bb[:, 3]
    w2, h2 = priors[:, 0], priors[:, 1]
    # overlap, assumes top left corner of both at (0, 0)
    horizontal_overlap = np.minimum(w1[:, None], w2)
    vertical_overlap = np.minimum(h1[:, None], h2)
    intersection = horizontal_overlap * vertical_overlap
    union = (w1 * h1)[:, None] + (w2 * h2) - intersection
    iou = intersection / union
    return np.argmax(iou, axis=1)
def processGroundTruth(bb, labels, priors, network_output_shape):
    """
    Given bounding boxes in normal x1,y1,x2,y2 format, the relevant labels in one-hot form,
    the anchor priors and the yolo model's output shape
    build the y_true vector to be used in yolov2 loss calculation
    """
    bb = bbToYoloFormat(bb) / 32
    best_anchor_indices = findBestPrior(bb, priors)
    responsible_grid_coords = np.floor(bb).astype(np.uint32)[:, :2]
    values = np.concatenate((
        bb, np.ones((len(bb), 1)), labels
    ), axis=1)
    x, y = np.split(responsible_grid_coords, 2, axis=1)
    y = y.ravel()
    x = x.ravel()
    y_true = np.zeros(network_output_shape)    
    y_true[y, x, best_anchor_indices] = values
    return y_true

class det_gen(tensorflow.keras.utils.Sequence):
    'Generates data from a Dataframe'
    def __init__(self,csv_file,patientId , img_path ,batch_size=8, dim=(256,256), n_channels=3,
                  shuffle=True, preprocess = None):

        self.df = csv_file
        self.shuffle = shuffle
        self.img_path = img_path
        self.patient_ids = patientId
        self.batch_size = batch_size
        self.nb_iteration = int(len(self.patient_ids)/self.batch_size)
        self.dim = dim
        self.n_channels= n_channels
        self.preprocess =preprocess
        self.TINY_YOLOV2_ANCHOR_PRIORS = np.array([1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]).reshape(5, 2)
        self.network_output_shape = (8,8,5,6)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.nb_iteration

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            random.shuffle(self.patient_ids)

    def __getitem__(self, index):
        'Generate one batch of data'
        indicies = range(index*self.batch_size, min((index*self.batch_size)+self.batch_size ,len(self.patient_ids) ))
        patientIds = self.patient_ids[indicies]
        X = np.zeros((self.batch_size, self.dim[0], self.dim[1],self.n_channels))
        y_boxes = []
        y = np.zeros((self.batch_size,self.network_output_shape[0],self.network_output_shape[1],self.network_output_shape[2],self.network_output_shape[3]))
        output_labels = []
        for index , patientId in enumerate(patientIds):
            filtered_df = self.df[self.df["patientId"] == patientId]
            img_path = os.path.join(self.img_path,patientId+".dcm" )
            img = self.load_img(img_path)
            X[index]= img
            y_boxes = []
            labels = []
            for i, row in filtered_df.iterrows():
                xmin = int(row['x'])
                ymin = int(row['y'])
                xmax = int(xmin + row['width'])
                ymax = int(ymin + row['height'])
                xmin = int((xmin/1024)*self.dim[0])
                xmax = int((xmax/1024)*self.dim[0])
                ymin = int((ymin/1024)*self.dim[1])
                ymax = int((ymax/1024)*self.dim[1])
                y_boxes.append([xmin,ymin,xmax,ymax])
                labels.append([1])
            #run preprocess_bboxes
            y[index] = processGroundTruth(np.array(y_boxes),np.array(labels), self.TINY_YOLOV2_ANCHOR_PRIORS , self.network_output_shape)
        return X, y

    def load_img(self,img_path):
        dcm_data = pydicom.read_file(img_path)
        a = dcm_data.pixel_array
        a=cv2.resize(a,(self.dim))
        if self.n_channels == 3:
            a = cv2.cvtColor(np.array(a, dtype=np.uint8), cv2.COLOR_GRAY2RGB)

        if self.preprocess != None:
            a= self.preprocess(a)

        a = exposure.equalize_adapthist(a)

        return a


def get_train_validation_generator(csv_path,img_path ,batch_size=8, dim=(256,256), n_channels=3,
                  shuffle=True ,preprocess = None , only_positive=True, validation_split=0.2 ):


  df = pd.read_csv(csv_path)
  if only_positive:
    df = df[df["Target"] == 1]

  random.seed(42)
  patient_ids = df["patientId"].unique()
  random.shuffle(patient_ids)

  patient_ids_train = patient_ids[int(len(patient_ids)*validation_split ):]
  patient_ids_validation = patient_ids[: int(len(patient_ids)*validation_split)]

  train_gen = det_gen(df,patient_ids_train , img_path ,batch_size=batch_size, dim=dim, n_channels=n_channels,
                shuffle=shuffle, preprocess = preprocess)

  validation_gen = det_gen(df, patient_ids_validation, img_path, batch_size=batch_size, dim=dim, n_channels=n_channels,
                       shuffle=shuffle, preprocess=preprocess)

  return train_gen, validation_gen


