import numpy as np
from sklearn.metrics import confusion_matrix

def compute_metrics_conf_matrix(logits,labels):
    '''
    logits.shape (Batchsize)
    labels.shape (Batchsize)
    '''
    logits = np.round(logits)
    corr = logits==labels
    incorr = ~corr 
    conf_matrix = confusion_matrix(labels,logits)
    return {
        'Corr': sum(corr),
        'Incorr': sum(incorr),
        'Total': len(labels),
        'Confmatrix': conf_matrix
    }

