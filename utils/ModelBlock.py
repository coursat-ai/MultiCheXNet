from tensorflow.keras.models import Model

class ModelBlock():
    def __init__():
        pass

    def load_weights(weight_path,load_from_start=True):
        """
        This function loads the weights to the model
        either take the weights from start pr from the end

        :param weight_path: string
            either path to model weights or model name like "ImageNet" if this is the encoder

        :param load_from_start: bool
            This boolean either loads the weights from start to finsh or till the dimentions doesn't match
            or from the last layer to the first layer or till the dimentions doesn't match
        """

        pass

    @staticmethod
    def add_heads(encoder, model_heads,is_classes=True):
        """
        This function adds a list of heads to the encoder block, a head can be
        classifcation, detection or segmentation head

        :param encoder: keras model.
            the head model that will serve as the encoder model for the Mulinet model

        :param model_heads: list.
            A list of keras models , each model will serve as a head on top of the encoder

        :return:
            combined model keras model.
        """


        model = Model(
            inputs=encoder.model.inputs,
            outputs=[model_head.model if is_classes else model_head  for model_head in model_heads] ,
            name="MultiCheXNet")
        return model

    @staticmethod
    def get_head_num_layers(encoder, model_head):
        return len(ModelBlock.add_heads(encoder, [model_head] , is_classes=False).layers) - len(encoder.model.layers)

