class DataGenerator(Sequence):
    def __init__(self, train_data_paths, csv_datafile, batch_size=32,
                 img_size=(256,256), transform=None, debug= False, shuffle=True, n_channels = 1):

        self.traindata_paths = train_data_paths
        self.stage2_annotations = csv_datafile
        self.batch_size = 32
        self.img_size = img_size
        self.n_channels = n_channels
        self.transform = transform
        self.shuffle = shuffle
        self.debug = debug
        self.clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
        self.on_epoch_end()

    def on_epoch_end(self):
        '''
         Updates indexes after each epoch'''
        self.indexes = np.arange(len(self.traindata_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        '''Denotes the number of batches per epoch
        returns:-
          number of batches per epoch'''
        return int(np.floor(len(self.traindata_paths) / self.batch_size))

    def __getitem__(self, index):
        '''Generate one batch of data
        args:-
            index: index of the batch
        returns:
            imgs and masks when fitting, imgs only when predicting'''

        #Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        #Find the list of patientIds
        list_IDs_batch = [self.traindata_paths[k] for k in indexes]

        imgs, masks = self.__data_generation(list_IDs_batch)

        if self.transform is None:
            if self.debug:
                return list_IDs_batch, imgs/255,masks/255
            else:
                return imgs/255.0,masks/255.0
        else:
            aug_imgs,aug_masks = [],[]
            for img,msk in zip(imgs,masks):
                augmented = self.transform(image=img, mask=msk)
                aug_imgs.append(augmented['image'])
                aug_masks.append(augmented['mask'])
            if self.debug:
                return list_IDs_batch, np.array(aug_imgs)/255, np.array(aug_masks)/255
            else:
                return np.array(aug_imgs)/255, np.array(aug_masks)/255


    def __data_generation(self, list_ID_batch):
        imgs = np.empty((self.batch_size, *self.img_size, self.n_channels))
        masks = np.empty((self.batch_size, *self.img_size, 1))

        for i, ID in enumerate(list_ID_batch):
            imgs[i,]= self.get_img(ID, self.img_size)
            masks[i,]= self.get_mask(ID, self.img_size, self.stage2_annotations)
        return imgs, masks


    def get_img(self, img_path, img_size):
        dicom_data = pydicom.dcmread(img_path)
        image = dicom_data.pixel_array
        img = cv2.resize(image, img_size, cv2.INTER_LINEAR)
        clahe_img = self.clahe.apply(img)
        clahe_img = np.expand_dims(clahe_img, axis=2)

        return clahe_img

    def get_mask(self, img_path, img_size, stage2_annotations):
        patient_id = img_path.split('/')[-1][:-4]
        rle = stage2_annotations.loc[stage2_annotations['ImageId'] == patient_id]['EncodedPixels'].values[0]
        if rle != '-1':
            mask = rle2mask(rle, 1024, 1024).T
            mask = cv2.resize(mask, img_size, cv2.INTER_LINEAR)
            mask = np.expand_dims(mask, axis=2)
        else :
            mask = np.zeros((img_size[0], img_size[1], 1))
        return mask
