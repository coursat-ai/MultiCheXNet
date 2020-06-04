from ModelBlock import ModelBlock

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense


class Classifier(ModelBlock):
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = self.make_model()

    def make_model(self):
        """
        This model is responsible for building a keras model

        :return:
            keras model:
        """
        model = Sequential()
        model.add(Input(shape=self.input_dim))
        model.add(Flatten())
        model.add(Dense(1000, activation='relu' ))
        model.add(Dense(12 , activation='softmax'))

        return model




