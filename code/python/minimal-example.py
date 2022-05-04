from sklearn.metrics import *

A = [0,0,1,1]
B = [1,1,0,2]

# sklearn.metrics
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
average = 'micro'  # 'micro', 'macro', 'samples','weighted', 'binary', None
print('NMI =', normalized_mutual_info_score(A, B))
print('AMI =', adjusted_mutual_info_score(A, B))
print('ARI =', adjusted_rand_score(A, B))
print('F1 =', f1_score(A, B, average=average))
print('F2 =', fbeta_score(A, B, beta=2, average=average))
print('FM =', fowlkes_mallows_score(A, B))
print('JI =', jaccard_score(A, B, average=average))
print('V =', v_measure_score(A, B))


import networkx as nx
g = nx.complete_bipartite_graph(2,2)

print('phi =', nx.conductance(g, A, B))  # ?
partition = [{0,1,2},{3}]
print('gamma =',nx.algorithms.community.quality.coverage(g, partition))

