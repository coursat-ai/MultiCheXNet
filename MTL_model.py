from .utils.ModelBlock import ModelBlock
from .utils.Encoder import Encoder
from .utils.Detector import Detector
from .utils.Segmenter import Segmenter
from .utils.Classifier import Classifier
from .utils.loss.MTL_loss import MTL_loss

class MTL_model():
    def __init__(self,dim=(256,256),add_class_head=True,add_detector_head=True,
                 add_segmenter_head=True):
        img_size = 256
        n_classes = 1
        self.encoder = Encoder(weights=None)
        if add_class_head:
            self.classifier = Classifier(self.encoder)
        if add_detector_head:
            self.detector = Detector(self.encoder, img_size, n_classes)
        if add_segmenter_head:
            self.segmenter = Segmenter(self.encoder)
        self.MTL_model = ModelBlock.add_heads(self.encoder, [self.classifier, self.detector, self.segmenter])


    def get_MTL_loss(self,classification_loss=None,detector_loss=None,segmenter_loss=None):

        if classification_loss!=None:
            classification_loss = classification_loss
        else:
            classification_loss = "categorical_crossentropy"

        if detector_loss!=None:
            detector_loss= detector_loss
        else:
            detector_loss = self.detector.loss

        if segmenter_loss!=None:
            segmenter_loss = segmenter_loss
        else:
            segmenter_loss = self.segmenter.loss

        mtl_loss = MTL_loss(classification_loss, detector_loss, segmenter_loss)

        return mtl_loss