import numpy as np
from sklearn.metrics import confusion_matrix

def compute_metrics_func(logits,labels):
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


def test_compute_metrics_func():
    def assert_equal(actual, expected, message):
        assert actual == expected, f"{message}. Expected: {expected}, Actual: {actual}"

    logits_sparse = np.random.rand(10, 5, 10)
    labels_sparse = np.full((10, 5), -100)
    for i in range(labels_sparse.shape[0]):
        j = np.random.randint(labels_sparse.shape[1])
        labels_sparse[i, j] = np.argmax(logits_sparse[i, j])
    metrics_sparse = compute_metrics_func(logits_sparse, labels_sparse)
    print(metrics_sparse)
    assert 0 <= metrics_sparse["hits@1"] <= 1, "Sparse Labels Test Case"
    assert 0 <= metrics_sparse["hits@10"] <= 1, "Sparse Labels Test Case"
    assert 0 <= metrics_sparse["MRR"] <= 1, "Sparse Labels Test Case"
    assert 0 <= metrics_sparse["MAP"] <= 1, "Sparse Labels Test Case"

    print("All test cases passed!")

#test_compute_metrics_func()
