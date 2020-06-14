# https://www.kaggle.com/cpmpml/fast-iou-metric-in-numpy-and-tensorflow
from keras import backend as K
def get_iou_vector(A, B):
    # Numpy version
    B = K.cast(B, 'float32')
    batch_size = A.shape[0]
    if batch_size is None:
      batch_size = 0
    metric = 0.0
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        true = np.sum(t)
        pred = np.sum(p)

        # deal with empty mask first
        if true == 0:
            pred_batch_size = pred / ( p.shape[0] * p.shape[1] )
            if pred_batch_size > 0.03:
               pred_batch_size = 1 
            metric +=  1 - pred_batch_size
            continue
        
        # non empty mask case.  Union is never empty 
        # hence it is safe to divide by its number of pixels

        intersection = np.sum(t * p)
        union = true + pred - intersection
        iou = intersection / union
        
        # iou metrric is a stepwise approximation of the real iou over 0.5
        iou = np.floor(max(0, (iou - 0.45)*20)) / 10
        
        metric += iou
        
    # teake the average over all images in batch
    metric /= batch_size
    return metric


def my_iou_metric(label, pred):
    # Tensorflow version
    return tf.py_function(get_iou_vector, [label, pred > 0.5], tf.float64)
    
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return 100 * score
