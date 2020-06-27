import numpy as np
from .classification_evaluation import evaluate_classification
from .detection_evaluation import evaluate_detection
from .segmentation_evaluation import evaluate_segmentation




def get_predictions_GT_MTL(predictions ,Y, Ground_truth_dict ,predictions_dict  ):

    if not (Y[0 ]==-1).all():
        if np.isnan(Ground_truth_dict["classification"]).all():
            Ground_truth_dict["classification" ]= Y[0]
        else:
            Ground_truth_dict["classification" ]= np.append(Ground_truth_dict["classification"] ,Y[0] ,axis=0)

        if np.isnan(predictions_dict["classification"]).all():
            predictions_dict["classification" ]= predictions[0]
        else:
            predictions_dict["classification" ]= np.append(predictions_dict["classification"] ,predictions[0] ,axis=0)

    if not (Y[1 ]==-1).all():
        if np.isnan(Ground_truth_dict["detection"]).all():
            Ground_truth_dict["detection" ]= Y[1]
        else:
            Ground_truth_dict["detection" ]= np.append(Ground_truth_dict["detection"] ,Y[1] ,axis=0)

        if np.isnan(predictions_dict["detection"]).all():
            predictions_dict["detection" ]= predictions[1]
        else:
            predictions_dict["detection" ]= np.append(predictions_dict["detection"] ,predictions[1] ,axis=0)

    if not (Y[2 ]==-1).all():
        if np.isnan(Ground_truth_dict["segmentation"]).all():
            Ground_truth_dict["segmentation" ]= Y[2]
        else:
            Ground_truth_dict["segmentation" ]= np.append(Ground_truth_dict["segmentation"] ,Y[2] ,axis=0)

        if np.isnan(predictions_dict["segmentation"]).all():
            predictions_dict["segmentation" ]= predictions[2]
        else:
            predictions_dict["segmentation" ]= np.append(predictions_dict["segmentation"] ,predictions[2] ,axis=0)

    return Ground_truth_dict ,predictions_dict

def get_predictions_GT_single_head(predictions ,Y, Ground_truth_dict ,predictions_dict,model_type='detector'):
    output_name=""
    if model_type == 'detector':
        model_type= "detection"
        
    if model_type == 'segmenter':
        model_type= "segmentation"
    
    if model_type == 'classification':
        model_type= "classification"
        
    if np.isnan(Ground_truth_dict[model_type]).all():
        Ground_truth_dict[model_type]= Y
    else:
        Ground_truth_dict[model_type ]= np.append(Ground_truth_dict[model_type] ,Y ,axis=0)

    if np.isnan(predictions_dict[model_type]).all():
        predictions_dict[model_type ]= predictions
    else:
        predictions_dict[model_type ]= np.append(predictions_dict[model_type] ,predictions ,axis=0)
        
    return Ground_truth_dict,predictions_dict
    
def evaluate(val_gen,model , model_type="" , anchors=None):
    
    """
    val_gen: keras data generator.
        This is the validation set data generator
    model: keras model
        is the keras model you want to evaluate
    model_type: string
        This string describes the model you passed.It can be one of the following
        'MTL': The full MTL model
        'detector': Encoder+detector head
        'segmenter': Encoder+segmenter head
        'classifier': Encoder+classifier head
    
    anchors: numpy array
        provided only if the model type is 'MTL' or 'detector'
        This is the detection anchor boxes used in training.
    """
    val_gen_itterator = val_gen.__iter__()

    Ground_truth_dict = {
        "classification": np.nan,
        "detection": np.nan,
        "segmentation": np.nan
    }

    predictions_dict = {
        "classification": np.nan,
        "detection": np.nan,
        "segmentation": np.nan
    }

    for index, (X, Y) in enumerate(val_gen_itterator):
        predictions = model.predict(X)
        if model_type=='MTL':
            Ground_truth_dict, predictions_dict = get_predictions_GT_MTL(predictions, Y, Ground_truth_dict, predictions_dict)
        else:
            Ground_truth_dict, predictions_dict = get_predictions_GT_single_head(predictions, Y, Ground_truth_dict, predictions_dict,model_type)
            
    
    if model_type=="MTL":
        mAP = evaluate_detection(Ground_truth_dict["detection"], predictions_dict['detection'],anchors)
        dice_coef = evaluate_segmentation(Ground_truth_dict["segmentation"], predictions_dict['segmentation'])
        f1,per,rec,acc = evaluate_classification(Ground_truth_dict["classification"], predictions_dict['classification'])
        return (mAP,dice_coef,f1,per,rec,acc)
    
    if model_type=="detector":
        mAP = evaluate_detection(Ground_truth_dict["detection"], predictions_dict['detection'],anchors)
        return mAP
    
    if model_type=="segmenter":
        dice_coef = evaluate_segmentation(Ground_truth_dict["segmentation"], predictions_dict['segmentation'])
        return dice_coef
    
    if model_type=="classifier":
        f1,per,rec,acc = evaluate_classification(Ground_truth_dict["classification"], predictions_dict['classification'])
        return (f1,per,rec,acc)
