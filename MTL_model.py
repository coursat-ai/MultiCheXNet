from .utils.ModelBlock import ModelBlock
from .utils.Encoder import Encoder
from .utils.Detector import Detector
from .utils.Segmenter import Segmenter
from .utils.Classifier import Classifier

class MTL_model():
    def __init__(self,dim=(256,256)):

        img_size = 256
        n_classes = 1

        self.encoder = Encoder(weights=None)
        self.classifier = Classifier(self.encoder)
        self.detector = Detector(self.encoder, img_size, n_classes)
        self.segmenter = Segmenter(self.encoder)
        self.MTL_model = ModelBlock.add_heads(self.encoder, [self.classifier, self.detector, self.segmenter])

    
