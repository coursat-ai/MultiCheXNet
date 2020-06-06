from ModelBlock import ModelBlock

class Detector(ModelBlock):
    def __init__(self):
        self.model = self.make_model()
        pass

    def make_model(self):
        """
        This model is responsible for building a keras model

        :return:
            keras model:
        """
        pass