class DataGenerator(keras.utils.Sequence):
  def __init__(self,data_csv, img_dir, batch_size=32, dim=(256,256),
               n_classes=2, transform=None, n_channels=1, shuffle=True,
               classes ={'Normal':0,'No Lung Opacity / Not Normal':1,'Lung Opacity': 2}, debug = False):
    '''Initialization
    args:-
      data_csv: data csv contains all the information of the patients
      img_dir: directory that contains the images
      batch_size: value of the batch size
      dim: tuple that contains the dimension of the generated images
      n_channels: value of the number of the channels of the images to be generated
      n_classes: if equal to 2 it will generate based on `Target` column,
                 and if equal to 3 it will generate based on `class` column
      self.classes: dict that map every categorical class to numerical value
      transform: Any list of transformation to be applied on the data (Augmentation)
      shuffle: bolean value if set to True, the data will be shuffled at the start of every epoch
    '''
    self.data_csv = data_csv
    self.patientIds = self.data_csv['patientId']
    self.img_dir = img_dir
    self.batch_size = batch_size
    self.dim = dim
    self.n_channels = n_channels
    self.n_classes = n_classes
    if n_classes == 2:
      self.classes = classes ={'Normal':0,'Lung Opacity': 1}
    else:
      self.classes = classes
    self.shuffle = shuffle
    self.debug = debug
    self.transform = transform
    self.on_epoch_end()

  def __len__(self):
    '''Denotes the number of batches per epoch
        returns:-
          number of batches per epoch'''
    return int(np.floor(len(self.data_csv) / self.batch_size))

  def __getitem__(self,index):
    '''Generate one batch of data
      args:-
        index: index of the batch
      returns:
        imgs and labels when fitting, imgs only when predicting'''
    #Generate indexes of the batch
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

    #Find the list of patientIds
    list_IDs_batch = [self.patientIds[k] for k in indexes]

    imgs, labels = self.__data_generation(list_IDs_batch, self.n_classes)

    if self.debug:
      return list_IDs_batch, imgs, labels
    else :
      return imgs, labels

  #maybe the same images
  def on_epoch_end(self):
    '''
    Updates indexes after each epoch
      '''
    self.indexes = np.arange(len(self.data_csv))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def __data_generation(self, list_IDs_batch, n_classes):
    '''
    Generates data containing batch_size samples
    args:-
        list_Ids_batch: lsit of the patientIds of specific batch number
    return:-
        imgs: An array of the patients images with shape (n_samples, *dim, n_channels) == 4D tensor
        labels: An array of encoded labels'''
    imgs = np.empty((self.batch_size, *self.dim, self.n_channels))
    labels = np.empty((self.batch_size), dtype=int)

    for i, ID in enumerate(list_IDs_batch):
      #Store patient image
      imgs[i,] = self.get_img(self.img_dir, ID, self.dim)
      labels[i] = self.get_label(ID, self.n_classes)
    if n_classes ==3:
      labels = keras.utils.to_categorical(labels, num_classes= n_classes) #one hot encoded if # of classes3
    return imgs, labels


  def get_img(self, img_dir, patientId,dim):
    '''
    helper function used to load patient image by `patientID`
    '''
    dcm_data = img_dir+'/%s.dcm' % patientId
    dcm_data = pydicom.read_file(dcm_data)
    img = dcm_data.pixel_array
    img = cv2.resize(img,dim)
    img = img.reshape((dim[0],dim[1],1))
    if not self.transform == None:
      params = self.transform.get_random_transform(img.shape)
      img = self.transform.apply_transform(img,params)
    else:
      img = np.float32(img) / 255.
    return img

  def get_label(self,patientId, n_classes):
    '''
    helper function used to get label specified label of the patient
    based on number of classes it will return the label of the patient
    '''
    full_info = train_data.loc[train_data['patientId'] == patientId].values
    if n_classes == 2:

      return int(full_info[0][-2]) #get the corresponding numerical value (0,1)

    elif n_classes == 3:
      class_label = full_info[:,-1] #get class type (e.g Normal, Lung Opacity)
      class_string = class_label[0]
      return int(self.classes[class_string]) #get the corresponding numerical value (0,1,2)
