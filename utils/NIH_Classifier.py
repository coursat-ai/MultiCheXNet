from .ModelBlock import ModelBlock
from tensorflow.keras.layers import *
from tensorflow.keras.losses import BinaryCrossentropy


class Classifier(ModelBlock):
    def __init__(self, encoder):
        self.encoder_output = encoder.model.output
        self.model = self.make_model()
        self.num_layers = ModelBlock.get_head_num_layers(encoder, self.model)

    def make_model(self):
        """
        This model is responsible for building a keras model that can classify ChestXRay data-set.
        :return:
            keras model:
        """

        x = GlobalAveragePooling2D()(self.encoder_output)

        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)

        output_layer = Dense(15, activation='sigmoid')(x)

        return output_layer

    def loss(self, y_true, y_pred):
        return BinaryCrossentropy(y_true, y_pred)
