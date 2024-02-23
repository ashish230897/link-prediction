import numpy as np

def compute_metrics_func(logits,labels):
    '''
    logits.shape (Batchsize, seq len, vocab)
    labels.shape (Batchsize, seq len)
    '''
    indices = np.where(labels != -100)
    logits = logits[indices]
    labels = labels[indices]
    top_predictions = np.argsort(logits, axis=1)[:, ::-1] 

    hits_at_1 = np.sum(top_predictions[:, 0] == labels)/len(labels)
    hits_at_10 = 0
    for i in range(len(labels)):
        if labels[i] in top_predictions[i][:10]:
            hits_at_10 += 1
    hits_at_10 = hits_at_10 / len(labels)

    reciprocal_ranks = []
    for i in range(len(labels)):
        rank = np.where(top_predictions[i] == labels[i])[0]
        if len(rank) > 0:
            reciprocal_rank = 1.0 / (rank[0] + 1)
            reciprocal_ranks.append(reciprocal_rank)
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    return {"hits@1": hits_at_1, "hits@10": hits_at_10, "MRR": mrr}

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
