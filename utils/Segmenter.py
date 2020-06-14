from .ModelBlock import ModelBlock
from tensorflow.keras.layers import *

class Segmenter(ModelBlock):
    def __init__(self, encoder):
        self.encoder = encoder
        self.encoder_output = encoder.output
        self.model = self.make_model()
        self.num_layers = ModelBlock.get_head_num_layers(encoder, self.model)

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
            x = conv_block(x, 32, name=name + '_block' + str(i + 1))
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
        tu5 = transition_up(db5, encoder, skip_layers[0], 3)

        db6 = dense_block(tu5, blocks[-1], name='conv6')
        tu6 = transition_up(db6, skip_layers[1], 3)

        db7 = dense_block(tu6, blocks[-2], name='conv7')
        tu7 = transition_up(db7, skip_layers[2], 3)

        db8 = dense_block(tu7, blocks[-3], name='conv8')
        tu8 = Conv2DTranspose(256, (3, 3), strides = (2, 2), padding = 'same')(db8)#(128,128,)

        uconv9 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(tu8)
        tu9 = Conv2DTranspose(256, (3, 3), strides = (2, 2), padding = 'same')(uconv9)#(256,256,)
        outputs = Conv2D(1, (1, 1), activation = 'sigmoid')(tu9)
        return outputs
        
