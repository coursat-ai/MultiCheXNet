from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow
import pandas as pd
import numpy as np
import cv2
from .text_cleaner import normalize_text

class det_gen(tensorflow.keras.utils.Sequence):
    'Generates data from a Dataframe'
    def __init__(self,df, tok, max_len,images_path, dim=(256,256), batch_size=8):
        self.df=df
        self.dim = dim
        self.images_path = images_path
        self.tok= tok
        self.max_len = max_len
        self.batch_size = batch_size
        self.nb_iteration = math.ceil((self.df.shape[0])/self.batch_size)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.nb_iteration

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.df=self.df.sample(frac=1)
    
    def load_img(self, img_path):
        
        img = cv2.imread(img_path)
        img =cv2.resize(img,(self.dim))
        
        return img
        
    
    def __getitem__(self, index):
        'Generate one batch of data'
        
        indicies = list(range(index*self.batch_size, min((index*self.batch_size)+self.batch_size ,(self.df.shape[0]))))
        
        images = []
        for img_path in self.df['filename'].iloc[indicies].tolist():
            img = self.load_img(os.path.join(self.images_path,img_path))
            images.append(img)
            
            
        
        
        x_batch = self.df['findings_cleaned'].iloc[indicies].tolist()
        
        x_batch_input = [sample[:-len(" endseq")] for sample in x_batch]
        
        x_batch_gt = [sample[len(" startseq"): ] for sample in x_batch]
        
        
        x_batch_input = np.array(pad_sequences( self.tok.texts_to_sequences (x_batch_input),
                          maxlen=self.max_len-1 ,
                          padding='post',
                          truncating='post'))
        
        x_batch_gt = np.array(pad_sequences( self.tok.texts_to_sequences (x_batch_gt),
                          maxlen=self.max_len-1 ,
                          padding='post',
                          truncating='post'))
        
        
        
        
        
        
        return [np.array(images), np.array(x_batch_input)] , np.array(x_batch_gt)   


def get_train_validation_generator(csv_path1,csv_path2,img_path, vocab_size,max_len,batch_size=8, dim=(256,256),shuffle=True ,preprocess = None , validation_split=0.2,augmentation=False,normalize=False,hist_eq =False,):
    
    df1= pd.read_csv(csv_path1)
    df2= pd.read_csv(csv_path2)
    
    df2 = df2[df2['projection']=='Frontal']
    
    df  =pd.merge(df1,df2,  on=['uid'])
    
    
    df= df.dropna(subset=['findings'])
    df['findings_cleaned'] = df['findings'].apply(normalize_text)
    
    
    vocab_size = vocab_size
    max_len = max_len
    tok = Tokenizer(num_words=vocab_size,  oov_token='UNK' )
    tok.fit_on_texts(df['findings_cleaned'].tolist())
    vocab_size = len(tok.word_index) + 1
    
    
    df = df.sample(frac=1,random_state=42)
    df_train = df.iloc[:-int(df.shape[0]*validation_split)]
    df_val   = df.iloc[-int(df.shape[0]*validation_split):]
    
    train_dataloader =  det_gen(df_train, tok, max_len,img_path)
    val_dataloader =  det_gen(df_val, tok, max_len,img_path)
    

    return train_dataloader, val_dataloader, vocab_size, tok
