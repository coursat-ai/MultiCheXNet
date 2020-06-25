import numpy as np

def dice_coef(y_true, y_pred, smooth=1):
    k = 1
    dice = np.sum(y_pred[y_true == k]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))
    return dice

def evaluate_segmentation(y_true, y_pred):
    dice_coef_score = []
    for GT, pred in zip(y_true, y_pred):
        diceCoef = dice_coef(GT, pred)
        dice_coef_score.append(diceCoef)
    return np.array(dice_coef_score).mean()