from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score,confusion_matrix
import numpy as np

def evaluate_classification(y_true, y_pred,binary=False,confusion_matrix_flag = True, classification_thresh=0.5):
    
    
    if binary:
      GT = y_true
      pred = (y_pred > classification_thresh)*1

      if confusion_matrix_flag:
          print(confusion_matrix(GT,pred))
      
      f1 = f1_score(GT,pred)
      per= precision_score(GT,pred)
      rec =recall_score(GT,pred)
      acc= accuracy_score(GT,pred)
      
    else:
      GT = np.argmax(y_true,axis=1)
      pred = np.argmax(y_pred,axis=1)
    
      if print_confusion_matrix:
          print(confusion_matrix(GT,pred))
      
      f1 = f1_score(GT,pred, average='micro')
      per= precision_score(GT,pred,average='micro')
      rec =recall_score(GT,pred,average='micro')
      acc= accuracy_score(GT,pred)


    return f1,per,rec,acc
