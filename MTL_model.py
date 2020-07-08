from .utils.ModelBlock import ModelBlock
from .utils.Encoder import Encoder
from .utils.Detector import Detector
from .utils.Segmenter import Segmenter
from .utils.Classifier import Classifier
from .utils.COVID_Classifier import COVID_Classifier

class MTL_model():
    def __init__(self,dim=(256,256),add_class_head=True,add_detector_head=True,
                 add_segmenter_head=True,classifier_type="MTL"):
        img_size = 256
        n_classes = 1
        self.encoder = Encoder(weights=None)
        self.encoder_num_layers = len(self.encoder.model.layers)
        self.add_class_head=add_class_head
        self.add_detector_head=add_detector_head
        self.add_segmenter_head=add_segmenter_head
        
        heads = []
        if self.add_class_head:
            if classifier_type=="MTL":
                self.classifier = Classifier(self.encoder)
                heads.append(self.classifier)
            if classifier_type == "COVID":
                self.classifier = COVID_Classifier(self.encoder)
                heads.append(self.classifier)
        if self.add_detector_head:
            self.detector = Detector(self.encoder, img_size, n_classes)
            heads.append(self.detector)
        if self.add_segmenter_head:
            self.segmenter = Segmenter(self.encoder)
            heads.append(self.segmenter)
            
        if self.add_class_head and self.add_detector_head and self.add_segmenter_head:
            self.classification_layers = [504,507,510,513,516]
            self.detector_layers = [505,508,511,514,517]
            self.segmenter_layers = sorted(list((set(range(427,519)) - set(self.classification_layers) - set(self.detector_layers))))
        
        self.MTL_model = ModelBlock.add_heads(self.encoder, heads)
        
        if int(self.add_class_head)+int(self.add_detector_head)+int(self.add_segmenter_head) ==1:
            if self.add_class_head:
                 self.classification_layers =list(range(self.encoder_num_layers,len(self.MTL_model.layers)))
            if self.add_detector_head:
                 self.detector_layers =list(range(self.encoder_num_layers,len(self.MTL_model.layers)))
            if self.add_segmenter_head:
                 self.segmenter_layers =list(range(self.encoder_num_layers,len(self.MTL_model.layers)))
        


    def get_MTL_loss(self,classification_loss=None,detector_loss=None,segmenter_loss=None):

        combined_losses = []
        if self.add_class_head:
            if classification_loss!=None:
                classification_loss = classification_loss
            else:
                classification_loss = self.classifier.loss
            combined_losses.append(classification_loss)

        if self.add_detector_head:
            if detector_loss!=None:
                detector_loss= detector_loss
            else:
                detector_loss = self.detector.loss
            combined_losses.append(detector_loss)

        if self.add_segmenter_head:
            if segmenter_loss!=None:
                segmenter_loss = segmenter_loss
            else:
                segmenter_loss = self.segmenter.loss
            combined_losses.append(segmenter_loss)

        if len(combined_losses)==1:
            combined_losses = combined_losses[0]

        return combined_losses
    
    
def load_weights(mtl_clss,weight_path, weight_part ,source, skip_end=0):
    """
    mtl_clss:
        MTL_class to load weights to

    weight_part:
      detector
      segmenter
      classifier
      encoder

    source:
      detector
      segmenter
      classifier
      MTL

      numpy
    """
    
    if source=='detector':
        other_model=MTL_model(add_class_head=False,add_detector_head=True,add_segmenter_head=False)
        other_model.MTL_model.load_weights(weight_path)
    elif source=='segmenter':
        other_model=MTL_model(add_class_head=False,add_detector_head=False,add_segmenter_head=True)
        other_model.MTL_model.load_weights(weight_path)
    elif source=='classifier':
        other_model=MTL_model(add_class_head=True,add_detector_head=False,add_segmenter_head=False)
        other_model.MTL_model.load_weights(weight_path)
    elif source== 'MTL':
        other_model=MTL_model()
        other_model.MTL_model.load_weights(weight_path)
    

    if source=='MTL':
        if weight_part=='detector':
          other_model_layers=other_model.detector_layers
          this_model_layers = mtl_clss.detector_layers
        if weight_part=='segmenter':
          other_model_layers=other_model.segmenter_layers
          this_model_layers=mtl_clss.segmenter_layers
        if weight_part=='classifier':
          other_model_layers=other_model.classification_layers
          this_model_layers= mtl_clss.classification_layers
    
    if weight_part=='encoder':
        other_model_layers = list(range(0,other_model.encoder_num_layers))
        this_model_layers = list(range(0,mtl_clss.encoder_num_layers))
    
    if skip_end>0:
        other_model_layers=other_model_layers[:-skip_end]
        this_model_layers=this_model_layers[:-skip_end]
    
    for index_to , index_from in zip(this_model_layers,other_model_layers):
        print(index_to,index_from)
        mtl_clss.MTL_model.layers[index_to].set_weights(other_model.MTL_model.layers[index_from].get_weights())

    return mtl_clss
