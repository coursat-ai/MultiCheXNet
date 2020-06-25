from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np

def evaluate_classification(y_true, y_pred):
    GT = np.argmax(y_true, axis=1)
    pred = np.argmax(y_pred, axis=1)

    f1 = f1_score(GT, pred)
    per = precision_score(GT, pred)
    rec = recall_score(GT, pred)
    acc = accuracy_score(GT, pred)

    return f1,per,rec,acc