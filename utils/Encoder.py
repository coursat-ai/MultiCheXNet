import tensorflow as tf

from tensorflow.keras.models import Model
from .ModelBlock import ModelBlock



class Encoder(ModelBlock):
    def __init__(self ,input_dim=(256,256,3) , model_type='densenet_121' , weights= "imagenet" ):
        self.input_dim=input_dim
        self.model_type = model_type
        self.weights=weights
        self.model = self.make_model()
        self.num_layers = len(self.model.layers)

    def make_model(self):
        """
        This model is responsible for building a keras model

        :return:
            keras model:
        """
        if self.model_type=='densenet_121':
            model = self.make_densenet_121(self.weights)

        return model

    def make_densenet_121(self,weights):
        model = tf.keras.applications.DenseNet121(
            include_top=False,
            weights=weights,
            input_tensor=None,
            input_shape=self.input_dim,
            pooling=None,
            classes=1000,
        )

        out = model.layers[426].output
        base_model = Model(model.input, outputs=out)

        return base_model