from keras import backend as K

#refrence: https://gist.github.com/wassname/7793e2058c5c9dacb5212c0ac0b18a8a
def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = y_true_f * y_pred_f
    score = (200. * K.sum(intersection) + smooth) / (100. * K.sum(y_true_f) + 100.* K.sum(y_pred_f) + smooth)
    return  (1. - score)

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))
