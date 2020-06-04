from tensorflow.keras.models import Sequential

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
    def add_head(encoder, model_heads):
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

        model = Sequential()
        model.add(encoder.model)
        for model_head in model_heads:
            model.add(model_head.model)

        return model