from .ModelBlock import ModelBlock
from tensorflow.keras.layers import Flatten,Input,Dense, GlobalAveragePooling2D , Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import categorical_crossentropy

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

        X = GlobalAveragePooling2D()(self.encoder_output)
        X = Dropout(0.2)(X)
        X = Dense(256, activation='relu' , activity_regularizer=l2(0.01))(X)
        X = Dropout(0.2)(X)
        X = Dense(3, activation='softmax')(X)

        return X
    def loss(self,y_true,y_pred):
        if tf.math.reduce_all(tf.math.equal(y_true,-1)):
            return  tf.convert_to_tensor(0, dtype=tf.float32)
        return categorical_crossentropy(y_true,y_pred)
