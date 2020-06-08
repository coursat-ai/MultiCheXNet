from .ModelBlock import ModelBlock
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense

class Detector(ModelBlock):
    def __init__(self, encoder):
        self.input_layer = encoder.model.output
        self.model = self.make_model()

    def make_model(self):
        """
        This model is responsible for building a keras model
        :return:
            keras model:
        """

        X = Flatten()(self.input_layer)
        X = Dense(3, activation='softmax')(X)

        return X