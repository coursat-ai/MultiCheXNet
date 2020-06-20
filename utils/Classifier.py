from .ModelBlock import ModelBlock

from tensorflow.keras.models import Sequential ,Flatten,Input,Dense GlobalAveragePooling1D , Dropout
from tensorflow.keras.regularizers import l2

class Classifier(ModelBlock):
    def __init__(self, encoder):
        self.encoder_output = encoder.model.output
        self.model = self.make_model()
        self.num_layers = ModelBlock.get_head_num_layers(encoder, self.model)

    def make_model(self):
        """
        This model is responsible for building a keras model
        :return:
            keras model:
        """

        X = GlobalAveragePooling1D()(self.encoder_output)
        X = Dropout(0.2)(X)
        X = Dense(256, activation='softmax' , activity_regularizer=l2(0.01))(X)
        X = Dropout(0.2)(X)
        X = Dense(3, activation='softmax')(X)

        return X
