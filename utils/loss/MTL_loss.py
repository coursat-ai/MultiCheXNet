def MTL_loss(classification_loss, detection_loss, segmentation_loss):
    return [classification_loss, detection_loss, segmentation_loss]