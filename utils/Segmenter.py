from .ModelBlock import ModelBlock
from tensorflow.keras.layers import *
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow.keras.backend as K
import tensorflow as tf

class Segmenter(ModelBlock):
    def __init__(self, encoder):
        self.encoder = encoder.model
        self.model = self.make_model()
        self.focal_loss= binary_focal_loss()
        #self.num_layers = ModelBlock.get_head_num_layers(encoder, self.model)

    def dense_block(self, x, blocks, name):
        #REF: keras-team
        """A dense block.
        # Arguments
            x: input tensor.
            blocks: integer, the number of building blocks.
            name: string, block label.
        # Returns
            output tensor for the block.
        """
        for i in range(blocks):
            x = self.conv_block(x, 32, name=name + '_block' + str(i + 1))
        return x

    def conv_block(self, x, growth_rate, name):
        #REF: keras-team
        """A building block for a dense block.
        # Arguments
            x: input tensor.
            growth_rate: float, growth rate at dense layers.
            name: string, block label.
        # Returns
            Output tensor for the block.
        """
        bn_axis = 3
        x1 = BatchNormalization(axis=bn_axis,
                                       epsilon=1.001e-5,
                                       name=name + '_0_bn')(x)
        x1 = Activation('relu', name=name + '_0_relu')(x1)
        x1 = Conv2D(4 * growth_rate, 1,
                           use_bias=False,
                           name=name + '_1_conv')(x1)
        x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                       name=name + '_1_bn')(x1)
        x1 = Activation('relu', name=name + '_1_relu')(x1)
        x1 = Conv2D(growth_rate, 3,
                           padding='same',
                           use_bias=False,
                           name=name + '_2_conv')(x1)
        x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
        return x

    def transition_up(self, x, skip_connection, out_channels, kernel_size=(3,3), stride=(2,2)):
        tu = Conv2DTranspose(out_channels, kernel_size, strides = stride, padding = 'same')(x)
        skip = self.encoder.layers[skip_connection].output
        c = concatenate([skip,tu], axis=3)
        return c

    def make_model(self, skip_layers=[308, 136, 48], blocks=[3, 3, 3, 3]):
        """
        This model is responsible for building a keras model
        :return:
            keras model:
        """
        db5 = self.encoder.output #(8,8,1024)
        tu5 = self.transition_up(db5, skip_layers[0], 3)

        db6 = self.dense_block(tu5, blocks[-1], name='conv6')
        tu6 = self.transition_up(db6, skip_layers[1], 3)

        db7 = self.dense_block(tu6, blocks[-2], name='conv7')
        tu7 = self.transition_up(db7, skip_layers[2], 3)

        db8 = self.dense_block(tu7, blocks[-3], name='conv8')
        tu8 = Conv2DTranspose(256, (3, 3), strides = (2, 2), padding = 'same')(db8)#(128,128,)

        uconv9 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(tu8)
        tu9 = Conv2DTranspose(256, (3, 3), strides = (2, 2), padding = 'same')(uconv9)#(256,256,)
        outputs = Conv2D(1, (1, 1), activation = 'sigmoid')(tu9)

        return outputs
    
    @tf.function
    def loss(self,y_true,y_pred):
        if tf.math.reduce_all(tf.math.equal(y_true,-1)):
            return  tf.convert_to_tensor(0, dtype=tf.float32)
        return self.focal_loss(y_true, y_pred)


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed
        
#refrence: https://gist.github.com/wassname/7793e2058c5c9dacb5212c0ac0b18a8a
def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = y_true_f * y_pred_f
    score = (200. * K.sum(intersection) + smooth) / (100. * K.sum(y_true_f) + 100.* K.sum(y_pred_f) + smooth)
    return  (1. - score)

def bce_dice_loss(y_true, y_pred):
    return BinaryCrossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def bce_logdice_loss(y_true, y_pred):
    return BinaryCrossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))
