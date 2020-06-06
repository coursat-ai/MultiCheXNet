
def MTL_loss(y_true,y_pred):
    overall_loss=0
    if y_true[0]!=None:
        #overall_loss+=classifcation loss
    if y_true[1]!=None:
        #overall_loss+=detection_loss
    if y_true[2]!=None:
        #overall_loss+=segmentation loss

    return overall_loss