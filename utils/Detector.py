from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Reshape
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.layers import  Reshape
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU
from tensorflow.keras import regularizers, initializers
import tensorflow as tf
from .ModelBlock import ModelBlock


def bbToYoloFormat(bb):
    """
    converts (left, top, right, bottom) to
    (center_x, center_y, center_w, center_h)
    """
    x1, y1, x2, y2 = np.split(bb, 4, axis=1)
    w = x2 - x1
    h = y2 - y1
    c_x = x1 + w / 2
    c_y = y1 + h / 2

    return np.concatenate([c_x, c_y, w, h], axis=-1)

def findBestPrior(bb, priors):
    """
    Given bounding boxes in yolo format and anchor priors
    compute the best anchor prior for each bounding box
    """
    w1, h1 = bb[:, 2], bb[:, 3]
    w2, h2 = priors[:, 0], priors[:, 1]

    # overlap, assumes top left corner of both at (0, 0)
    horizontal_overlap = np.minimum(w1[:, None], w2)
    vertical_overlap = np.minimum(h1[:, None], h2)

    intersection = horizontal_overlap * vertical_overlap
    union = (w1 * h1)[:, None] + (w2 * h2) - intersection
    iou = intersection / union
    return np.argmax(iou, axis=1)

def processGroundTruth(bb, labels, priors, network_output_shape):
    """
    Given bounding boxes in normal x1,y1,x2,y2 format, the relevant labels in one-hot form,
    the anchor priors and the yolo model's output shape
    build the y_true vector to be used in yolov2 loss calculation
    """
    bb = bbToYoloFormat(bb) / 32
    best_anchor_indices = findBestPrior(bb, priors)

    responsible_grid_coords = np.floor(bb).astype(np.uint32)[:, :2]

    values = np.concatenate((
        bb, np.ones((len(bb), 1)), labels
    ), axis=1)

    x, y = np.split(responsible_grid_coords, 2, axis=1)
    y = y.ravel()
    x = x.ravel()

    y_true = np.zeros(network_output_shape)

    y_true[y, x, best_anchor_indices] = values

    return y_true

def conv_batch_lrelu(input_tensor, numfilter, dim, strides=1):
    input_tensor = Conv2D(numfilter, (dim, dim), strides=strides, padding='same',
                        kernel_regularizer=regularizers.l2(0.0005),
                        kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                        use_bias=False
                    )(input_tensor)
    input_tensor = BatchNormalization()(input_tensor)
    return LeakyReLU(alpha=0.1)(input_tensor)

class Detector(ModelBlock):

    def __init__(self,encoder, image_size, n_classes):

        self.encoder_output = encoder.model.output

        self.TINY_YOLOV2_ANCHOR_PRIORS = np.array([
            1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52
        ]).reshape(5, 2)

        K.reset_uids()

        self.image_size = image_size
        self.n_cells = self.image_size // 32
        self.B = self.TINY_YOLOV2_ANCHOR_PRIORS.shape[0]
        self.n_classes = n_classes

        self.model = self.make_model()

        self.num_layers = ModelBlock.get_head_num_layers(encoder, self.model)



    def make_model(self):
        """
        This model is responsible for building a keras model
        :return:
            keras model:
        """

        model = conv_batch_lrelu(self.encoder_output, 1024, 3)

        n_outputs = len(self.TINY_YOLOV2_ANCHOR_PRIORS) * (5 + self.n_classes)
        model = Conv2D(n_outputs, (1, 1), padding='same', activation='linear')(model)

        model_out = Reshape(
            [self.n_cells, self.n_cells, self.B, 4 + 1 + self.n_classes]
        )(model)

        return model_out

    
    def loss(self, y_true, y_pred):
        n_cells = y_pred.get_shape().as_list()[1]
        y_true = tf.reshape(y_true, tf.shape(y_pred), name='y_true')
        y_pred = tf.identity(y_pred, name='y_pred')
        
        TINY_YOLOV2_ANCHOR_PRIORS = tf.convert_to_tensor(self.TINY_YOLOV2_ANCHOR_PRIORS, dtype= tf.float32)
        
        #### PROCESS PREDICTIONS ####
        # get x-y coords (for now they are with respect to cell)
        predicted_xy = tf.nn.sigmoid(y_pred[..., :2])

        # convert xy coords to be with respect to image
        cell_inds = tf.range(n_cells, dtype=tf.float32)
        predicted_xy = tf.stack((
            predicted_xy[..., 0] + tf.reshape(cell_inds, [1, -1, 1]),
            predicted_xy[..., 1] + tf.reshape(cell_inds, [-1, 1, 1])
        ), axis=-1)

        # compute bb width and height
        predicted_wh = TINY_YOLOV2_ANCHOR_PRIORS * tf.exp(y_pred[..., 2:4])

        # compute predicted bb center and width
        predicted_min = predicted_xy - predicted_wh / 2
        predicted_max = predicted_xy + predicted_wh / 2

        predicted_objectedness = tf.nn.sigmoid(y_pred[..., 4])
        predicted_logits = tf.nn.softmax(y_pred[..., 5:])

        #### PROCESS TRUE ####
        true_xy = y_true[..., :2]
        true_wh = y_true[..., 2:4]
        true_logits = y_true[..., 5:]

        true_min = true_xy - true_wh / 2
        true_max = true_xy + true_wh / 2

        #### compute iou between ground truth and predicted (used for objectedness) ####
        intersect_mins = tf.maximum(predicted_min, true_min)
        intersect_maxes = tf.minimum(predicted_max, true_max)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = predicted_wh[..., 0] * predicted_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = intersect_areas / union_areas

        #### Compute loss terms ####
        responsibility_selector = y_true[..., 4]

        xy_diff = tf.square(true_xy - predicted_xy) * responsibility_selector[..., None]
        xy_loss = tf.reduce_sum(xy_diff, axis=[1, 2, 3, 4])

        wh_diff = tf.square(tf.sqrt(true_wh) - tf.sqrt(predicted_wh)) * responsibility_selector[..., None]
        wh_loss = tf.reduce_sum(wh_diff, axis=[1, 2, 3, 4])

        obj_diff = tf.square(iou_scores - predicted_objectedness) * responsibility_selector
        obj_loss = tf.reduce_sum(obj_diff, axis=[1, 2, 3])

        best_iou = tf.reduce_max(iou_scores, axis=-1)
        no_obj_diff = tf.square(0 - predicted_objectedness) * tf.cast(best_iou < 0.6, dtype=tf.float32)[..., None] * (
                    1 - responsibility_selector)
        no_obj_loss = tf.reduce_sum(no_obj_diff, axis=[1, 2, 3])

        clf_diff = tf.square(true_logits - predicted_logits) * responsibility_selector[..., None]
        clf_loss = tf.reduce_sum(clf_diff, axis=[1, 2, 3, 4])

        object_coord_scale = 5
        object_conf_scale = 1
        noobject_conf_scale = 1
        object_class_scale = 1

        loss = object_coord_scale * (xy_loss + wh_loss) + \
               object_conf_scale * obj_loss + noobject_conf_scale * no_obj_loss + \
               object_class_scale * clf_loss
        return loss

