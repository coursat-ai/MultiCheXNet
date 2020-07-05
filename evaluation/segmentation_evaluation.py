import numpy as np

def dice_coef(y_true, y_pred, smooth=1):
    k=1
    dice =(np.sum(y_pred[y_true==k])*2.0 + smooth) / (np.sum(y_pred) + np.sum(y_true)+smooth)
    return  dice

def evaluate_segmentation(y_true,y_pred,y_class_pred=np.nan):
    dice_coef_list = []
    for index,(GT , pred) in enumerate(zip(y_true ,y_pred )):
        
        if not np.isnan(y_class_pred).any():
            if np.argmax(y_class_pred[index])==0 and GT.sum()==0:
                dd.append(1)
            else:
                diceCoef = dice_coef(GT , pred)
                dice_coef_list.append(diceCoef)
        else:
            diceCoef = dice_coef(GT , pred)
            dice_coef_list.append(diceCoef)
            
    return  np.array(dice_coef_list).mean()
