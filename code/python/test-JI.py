import numpy as np
from sklearn.metrics import jaccard_score
y_true = np.array([[0, 1, 1],[1,1,0]])
y_pred = np.array([[1, 1, 1],[1, 0, 0]])
print(y_true[0])
print(y_pred[0])
JI=jaccard_score(y_true[0], y_pred[0],average = 'micro')
print(JI)


from sklearn import metrics
labels_true = [0, 0, 0, 1, 1, 1]
labels_pred = [0, 0, 1, 1, 2, 2]
RS=metrics.rand_score(labels_true, labels_pred)
print(RS)
