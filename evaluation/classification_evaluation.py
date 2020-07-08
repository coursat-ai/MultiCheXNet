from sklearn.metrics import f1_score , precision_score , recall_score ,accuracy_score ,confusion_matrix

def evaluate_classification(y_true,y_pred, print_confusion_matrix= False):
    GT = np.argmax(y_true,axis=1)
    pred = np.argmax(y_pred,axis=1)
    
    if print_confusion_matrix:
        print(confusion_matrix(GT,pred))
    
    f1 = f1_score(GT,pred, average='micro')
    per= precision_score(GT,pred,average='micro')
    rec =recall_score(GT,pred,average='micro')
    acc= accuracy_score(GT,pred)
    return f1,per,rec,acc
