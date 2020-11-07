import tensorflow


import numpy as np
from .SIIM_ACR_dataloader import get_train_validation_generator as segmenatation_get_train_validation_generator
from .RSNA_dataloader import get_train_validation_generator as detection_get_train_validation_generator
from .indiana_dataloader import get_train_validation_generator as report_gen_get_train_validation_generator

class MTL_generatot(tensorflow.keras.utils.Sequence):
    'Generates data from a Dataframe'

    def __init__(self, Segmentation_gen, detection_gen , report_gen,  nb_iteration,batch_size,gen_type):
        'Initialization'

        self.seg_generator = Segmentation_gen
        self.det_generator = detection_gen
        self.report_gen = report_gen
        
        self.gen_type=gen_type
        self.first_flag=0
        
        self.seg_itterator = self.seg_generator.__iter__()
        self.det_itterator = self.det_generator.__iter__()
        if self.report_gen is not None:
            self.report_itterator = self.report_gen.__iter__()
        
        self.nb_iteration = nb_iteration
        self.batch_size = batch_size
        self.batch_number = 1

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
        if self.gen_type=='train':
            if self.batch_number == 1:
                try:
                    X, Y_seg = next(self.seg_itterator)
                except:
                    self.seg_itterator = self.seg_generator.__iter__()
                    X, Y_seg = next(self.seg_itterator)

                Y_class= []
                for yy in Y_seg:
                    if np.sum(yy)==0:
                        Y_class.append([1,0,0])
                    else:
                        Y_class.append([0, 1, 0])

                Y_class = np.array(Y_class)
                Y_seg= np.array(Y_seg)
                Y_det = np.ones([self.batch_size,8,8,5,6])*-1
                
                if self.report_gen is not None:
                    X_report = np.zeros([self.batch_size , self.report_gen.max_len-1])*-1
                    Y_report = np.ones([self.batch_size , self.report_gen.max_len-1])*-1

            if self.batch_number == 2:
                try:
                    X, Y_det = next(self.det_itterator)
                except:
                    self.det_itterator = self.det_generator.__iter__()
                    X, Y_det = next(self.det_itterator)

                Y_class= []
                for yy in Y_det:
                    if np.sum(yy)==0:
                        Y_class.append([1,0,0])
                    else:
                        Y_class.append([0, 0, 1])
                Y_class = np.array(Y_class)
                Y_det= np.array(Y_det)
                Y_seg = np.ones([self.batch_size,256,256,1])*-1
                
                if self.report_gen is not None:
                    X_report = np.zeros([self.batch_size , self.report_gen.max_len-1])*-1
                    Y_report = np.ones([self.batch_size , self.report_gen.max_len-1 ])*-1
            
                if self.report_gen is None:
                    self.batch_number =0 
                
            if self.batch_number == 3:
                try:
                    (X,X_report), Y_report = next(self.report_itterator) 
                except:
                    self.report_itterator = self.report_gen.__iter__()
                    (X,X_report), Y_report = next(self.report_itterator) 
                
                Y_class = np.ones([self.batch_size,3])*-1 
                Y_det = np.ones([self.batch_size,8,8,5,6])*-1
                Y_seg = np.ones([self.batch_size,256,256,1])*-1
                
                self.batch_number =0
            self.batch_number += 1
        
        
        if self.gen_type=='val':
            if self.first_flag==0:
                try:
                    X, Y_seg = next(self.seg_itterator)
                    Y_class= []
                    for yy in Y_seg:
                        if np.sum(yy)==0:
                            Y_class.append([1,0,0])
                        else:
                            Y_class.append([0, 1, 0])

                    Y_class = np.array(Y_class)
                    Y_seg= np.array(Y_seg)
                    Y_det = np.ones([self.batch_size,8,8,5,6])*-1
                except:
                    self.first_flag=1
                    self.seg_itterator = self.seg_generator.__iter__()
                    
            if self.first_flag==1:
                try:
                    X, Y_det = next(self.det_itterator)
                    Y_class= []
                    for yy in Y_det:
                        if np.sum(yy)==0:
                            Y_class.append([1,0,0])
                        else:
                            Y_class.append([0, 0, 1])
                    Y_class = np.array(Y_class)
                    Y_det= np.array(Y_det)
                    Y_seg = np.ones([self.batch_size,256,256,1])*-1
                except:
                    self.first_flag=0
                    self.det_itterator = self.det_generator.__iter__()
                    
        if self.report_gen is None:
            return X, [Y_class, Y_det, Y_seg]
        else:
            return [X,X_report], [Y_class, Y_det, Y_seg , Y_report]

def get_train_validation_generator(det_csv_path,seg_csv_path , det_img_path, seg_img_path, report_csv_path1=None,report_csv_path2=None,report_img_path=None, batch_size=8, dim=(256,256), n_channels=3,
                  shuffle=True ,preprocess = None , only_positive=True, validation_split=0.2,augmentation=False,normalize=False,hist_eq=False,batch_positive_portion=None ):
    


    seg_train_gen, seg_valid_gen = segmenatation_get_train_validation_generator(seg_csv_path, seg_img_path,
                                                                                batch_size=batch_size, dim=dim,
                                                                                n_channels=n_channels, shuffle=shuffle,
                                                                                preprocess=preprocess,
                                                                                only_positive=only_positive,
                                                                                augmentation=augmentation,
                                                                                normalize=normalize,
                                                                                hist_eq=hist_eq,
                                                                                validation_split=validation_split,batch_positive_portion=batch_positive_portion)

    det_train_gen, det_valid_gen = detection_get_train_validation_generator(det_csv_path, det_img_path,
                                                                                batch_size=batch_size, dim=dim,
                                                                                n_channels=n_channels, shuffle=shuffle,
                                                                                preprocess=preprocess,
                                                                                only_positive=only_positive,
                                                                                augmentation=augmentation,
                                                                                normalize=normalize,
                                                                                hist_eq=hist_eq,
                                                                                validation_split=validation_split,batch_positive_portion=batch_positive_portion)

    if report_csv_path1 is not None :
        vocab_size=10000
        max_len=100
        shuffle_GT_sentences=True
        feat_model=None
        report_train_gen, report_valid_gen, vocab_size, tok =  report_gen_get_train_validation_generator(report_csv_path1,report_csv_path2,
                                       report_img_path, vocab_size,max_len,
                                       batch_size=batch_size, dim=dim,
                                       shuffle=shuffle , preprocess = preprocess ,
                                       validation_split=validation_split,augmentation=augmentation,
                                       normalize=normalize,hist_eq =hist_eq,
                                       shuffle_GT_sentences=shuffle_GT_sentences ,
                                       feat_model=feat_model,over_sample=batch_positive_portion)
        
        MTL_train_gen = MTL_generatot(seg_train_gen, det_train_gen ,report_train_gen,
                                      seg_train_gen.nb_iteration+det_train_gen.nb_iteration +report_train_gen.nb_iteration, 
                                      batch_size=batch_size , gen_type='train')
        
        MTL_valid_gen = MTL_generatot(seg_valid_gen, det_valid_gen,report_valid_gen,
                                      seg_valid_gen.nb_iteration + det_valid_gen.nb_iteration+ report_valid_gen.nb_iteration,
                                      batch_size=batch_size, gen_type='val')
    
    else:
        MTL_train_gen = MTL_generatot(seg_train_gen, det_train_gen ,None, seg_train_gen.nb_iteration+det_train_gen.nb_iteration ,batch_size=batch_size , gen_type='train')
        MTL_valid_gen = MTL_generatot(seg_valid_gen, det_valid_gen, None, seg_valid_gen.nb_iteration + det_valid_gen.nb_iteration , batch_size=batch_size, gen_type='val')

    return MTL_train_gen, MTL_valid_gen


