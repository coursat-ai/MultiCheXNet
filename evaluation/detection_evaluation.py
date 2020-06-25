# https://github.com/guigzzz/Keras-Yolo-v2/blob/master/postprocessing.py
import numpy as np
import os
import shutil
import subprocess


def logistic(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_out = np.exp(x - np.max(x, axis=-1)[..., None])
    return exp_out / np.sum(exp_out, axis=-1)[..., None]


# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def non_max_suppression(boxes, labels, conf, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        i = idxs[-1]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:-1]])
        yy1 = np.maximum(y1[i], y1[idxs[:-1]])
        xx2 = np.minimum(x2[i], x2[idxs[:-1]])
        yy2 = np.minimum(y2[i], y2[idxs[:-1]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:-1]]

        # delete all indexes from the index list that have
        idxs = (idxs[:-1])[overlap < overlapThresh]

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick], labels[pick], conf[pick]


def getBoundingBoxesFromNetOutput(clf, anchors, confidence_threshold, cell_size, pred=True):
    pw, ph = anchors[:, 0], anchors[:, 1]
    cell_inds = np.arange(clf.shape[1])

    tx = clf[..., 0]
    ty = clf[..., 1]
    tw = clf[..., 2]
    th = clf[..., 3]
    to = clf[..., 4]

    sftmx = softmax(clf[..., 5:])
    predicted_labels = np.argmax(sftmx, axis=-1)
    class_confidences = np.max(sftmx, axis=-1)

    if pred:
        bx = logistic(tx) + cell_inds[None, :, None]
        by = logistic(ty) + cell_inds[:, None, None]
        bw = pw * np.exp(tw) / 2
        bh = ph * np.exp(th) / 2
        object_confidences = logistic(to)
    else:
        bx = tx  # logistic(tx) + cell_inds[None, :, None]
        by = ty  # logistic(ty) + cell_inds[:, None, None]
        bw = tw / 2  # pw * np.exp(tw) / 2
        bh = th / 2  # ph * np.exp(th) / 2
        object_confidences = to

    left = bx - bw
    right = bx + bw
    top = by - bh
    bottom = by + bh

    boxes = np.stack((
        left, top, right, bottom
    ), axis=-1) * cell_size

    final_confidence = class_confidences * object_confidences
    boxes = boxes[final_confidence > confidence_threshold].reshape(-1, 4).astype(np.int32)
    labels = predicted_labels[final_confidence > confidence_threshold]
    return boxes, labels, final_confidence[final_confidence > confidence_threshold]


def yoloPostProcess(yolo_output, priors, maxsuppression=True, maxsuppressionthresh=0.5, classthresh=0.6, cell_size=32,
                    pred=True):
    allboxes = []
    for o in yolo_output:
        boxes, labels, conf = getBoundingBoxesFromNetOutput(o, priors, confidence_threshold=classthresh,
                                                            cell_size=cell_size, pred=pred)
        if maxsuppression and len(boxes) > 0:
            boxes, labels, conf = non_max_suppression(boxes, labels, conf, maxsuppressionthresh)
        allboxes.append((boxes, labels, conf))
    return allboxes


def evaluate_detection(y_true, y_pred, anchor_boxes, classthresh=0.1, GT_path='mAP/input/ground-truth/',
                       pred_path='mAP/input/detection-results/'):
    # refrence https://github.com/Cartucho/mAP
    if not os.path.isdir('mAp'):
        os.system("git clone https://github.com/Cartucho/mAP.git")

    if os.path.isdir(GT_path):
        shutil.rmtree(GT_path)
    os.mkdir(GT_path)

    if os.path.isdir(pred_path):
        shutil.rmtree(pred_path)
    os.mkdir(pred_path)

    boxes_gt = yoloPostProcess(y_true, anchor_boxes, pred=False)
    boxes_pred = yoloPostProcess(y_pred, anchor_boxes, pred=True, classthresh=classthresh)

    for index, (gt, pred) in enumerate(zip(boxes_gt, boxes_pred)):
        gt_str = ""
        for (box, label, conf) in zip(gt[0], gt[1], gt[2]):
            gt_str += "Pneumonia " + str(box[0]) + " " + str(box[1]) + " " + str(box[2]) + " " + str(box[3]) + "\n"

        open(os.path.join(GT_path, str(index) + ".txt"), 'w').write(gt_str)

        pred_str = ""
        for (box, label, conf) in zip(pred[0], pred[1], pred[2]):
            pred_str += "Pneumonia " + str(conf) + " " + str(box[0]) + " " + str(box[1]) + " " + str(
                box[2]) + " " + str(box[3]) + "\n"

        open(os.path.join(pred_path, str(index) + ".txt"), 'w').write(pred_str)

    mAP_str = subprocess.check_output("cd mAP/;python3 main.py --no-plot -na", shell=True);
    mAP = str(mAP_str).split('mAP')[1].split('%')[0].split(' ')[-1]

    return float(mAP) / 100