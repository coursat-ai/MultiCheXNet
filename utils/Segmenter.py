from .ModelBlock import ModelBlock
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense

class Segmenter(ModelBlock):
    def __init__(self, encoder):
        self.encoder_output = encoder.model.output
        self.model = self.make_model()

    def make_model(self):
        """
        This model is responsible for building a keras model
        :return:
            keras model:
        """

        X = Flatten( name="seg_flatten")(self.encoder_output)
        X = Dense(3, activation='softmax' , name="seg_dense")(X)

        return X