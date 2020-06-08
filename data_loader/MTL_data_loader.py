import tensorflow


import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from .chex_xray_14 import Data_generator as Chex_gen
from .SIIM_ACR import Seg_Dataloader as Seg_gen

class MTL_generatot(tensorflow.keras.utils.Sequence):
    'Generates data from a Dataframe'

    def __init__(self, df_seg_path, seg_data_folder, df_det, batch_size=8, dim=(256, 256), shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.df_seg = df_seg_path
        # TF 100%
        self.df_seg = self.df_seg[self.df_seg['Class'] != 0].reset_index()
        # TF 100%
        self.df_det = df_det
        # self.df_det  = self.df_det[self.df_det['Class_Det']!=0].reset_index()

        self.seg_generator = Seg_gen.Seg_gen(self.df_seg,seg_data_folder, preprocess_input, batch_size=self.batch_size, dim=self.dim,
                                             shuffle=True)
        self.det_generator = Chex_gen.Data_gen(self.df_det, preprocess_input, batch_size=self.batch_size, dim=self.dim,
                                               shuffle=True)

        self.n = len(self.df_seg) + len(self.df_det)

        self.nb_iteration = int(np.floor(self.n / self.batch_size))
        self.on_epoch_end()
        self.batch_number = 0

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.nb_iteration

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.df_seg) + len(self.df_det))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization

        if self.batch_number % 2 == 0:
            X, Y_seg = next(iter(self.seg_generator))
            Y_class = [[0, 1, 0]] * self.batch_size
            Y_det = None

        else:
            X, Y_det = next(iter(self.det_generator))
            Y_class = [[0, 1, 0]] * self.batch_size
            Y_seg = None

        self.batch_number += 1

        return X, np.array([np.array(Y_class), np.array(Y_det), np.array(Y_seg)])