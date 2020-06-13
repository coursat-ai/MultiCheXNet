import tensorflow


import numpy as np
from .SIIM_ACR_dataloader import get_train_validation_generator as segmenatation_get_train_validation_generator
from .RSNA_dataloader import get_train_validation_generator as detection_get_train_validation_generator

class MTL_generatot(tensorflow.keras.utils.Sequence):
    'Generates data from a Dataframe'

    def __init__(self, Segmentation_gen, detection_gen , nb_iteration):
        'Initialization'

        self.seg_generator = Segmentation_gen
        self.det_generator = detection_gen
        self.nb_iteration = nb_iteration
        self.batch_number = 0

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.nb_iteration

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # Generate data
        X, y = self.__data_generation(index)

        return X, y

    def __data_generation(self, index):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization

        if self.batch_number % 2 == 0:
            X, Y_seg = next(iter(self.seg_generator))
            Y_class = [[0, 1, 0]] * self.batch_size
            Y_det = None

        else:
            X, Y_det = next(iter(self.det_generator))
            Y_class = [[0, 0, 1]] * self.batch_size
            Y_seg = None

        self.batch_number += 1

        return X, np.array([np.array(Y_class), np.array(Y_det), np.array(Y_seg)])

def get_train_validation_generator(det_csv_path,seg_csv_path , det_img_path, seg_img_path ,batch_size=8, dim=(256,256), n_channels=3,
                  shuffle=True ,preprocess = None , only_positive=True, validation_split=0.2 ):




    seg_train_gen, seg_valid_gen = segmenatation_get_train_validation_generator(seg_csv_path, seg_img_path,
                                                                                batch_size=batch_size, dim=dim,
                                                                                n_channels=n_channels, shuffle=shuffle,
                                                                                preprocess=preprocess,
                                                                                only_positive=only_positive,
                                                                                validation_split=validation_split)

    det_train_gen, det_valid_gen = detection_get_train_validation_generator(det_csv_path, det_img_path,
                                                                                batch_size=batch_size, dim=dim,
                                                                                n_channels=n_channels, shuffle=shuffle,
                                                                                preprocess=preprocess,
                                                                                only_positive=only_positive,
                                                                                validation_split=validation_split)

    MTL_train_gen = MTL_generatot(seg_train_gen, det_train_gen , seg_train_gen.nb_iteration+det_train_gen.nb_iteration)
    MTL_valid_gen = MTL_generatot(seg_valid_gen, det_valid_gen, seg_valid_gen.nb_iteration + det_valid_gen.nb_iteration)

    return MTL_train_gen, MTL_valid_gen


