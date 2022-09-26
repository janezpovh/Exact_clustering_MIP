from email import header
from pickle import TRUE
import igraph as ig
from sklearn.metrics import *
import os
import numpy as np
import pandas as pd

#podatki = np.loadtxt("C:\\Users\\janez\\Documents\\MATLAB\\jpcode\\exact_clustering_MIP\\results\\Optimum_clusterings\\Karate_k_2.txt",delimiter='\t',header=TRUE)

podatki_clustering = pd.read_csv("C:\\Users\\janez\\Documents\\MATLAB\\jpcode\\exact_clustering_MIP\\results\\Optimum_clusterings\\Karate_k_2.txt", sep="\t",  skip_blank_lines=True, header=0) #names=['Kmed','MSS','Kmed_sq','MSS_sq'])
print("podatki_clustering")
print(podatki_clustering)

podatki_graf = pd.read_csv("C:\\Users\\janez\\Documents\\MATLAB\\jpcode\\exact_clustering_MIP\\data\\networks\\karate.txt", sep="\t",  skip_blank_lines=True, header=0) #names=['Kmed','MSS','Kmed_sq','MSS_sq'])
print("podatki_graf")
print(podatki_graf)



os.chdir('C:\\Users\\janez\\Documents\\MATLAB\\jpcode\\exact_clustering_MIP\\results')

def conductance(g, c):
    """Evaluates the conductance of the clustering solution c on a given graph g."""
    n = len(g.vs)
    m = len(g.es)

    ASSc = {}
    AS = {}

    for col in set(c):
        ASSc[col] = 0
        AS[col] = 0

    for i in range(n):
        current_clr = c[i]
        for j in g.neighbors(i):
            if c[i] == c[j]:
                AS[current_clr] += 1 / 2
            else:
                ASSc[current_clr] += 1
                AS[current_clr] += 1

    phi_S = {}

    for col in set(c):
        if (min(AS[col], m - AS[col] + ASSc[col]) == 0):
            phi_S[col] = 0
        else:
            phi_S[col] = ASSc[col] / min(AS[col], m - AS[col] + ASSc[col])

    phi = 0

    for col in set(c):
        phi -= phi_S[col]

    phi /= len(set(c))
    phi += 1

    intra_cluster_phi = min(phi_S.values())
    inter_cluster_phi = 1 - max(phi_S.values())

    return phi, intra_cluster_phi, inter_cluster_phi


def coverage(g, c):
    """Evaluates the coverage of the clustering solution c on a given graph g."""
    n = len(g.vs)
    A_delta = 0
    A = 0

    for i in range(n):
        for j in g.neighbors(i):
            A += 1
            if c[i] == c[j]:
                A_delta += 1

    return A_delta / A

# C:\Users\janez\Documents\MATLAB\jpcode\exact_clustering_MIP\results\p-medians results


#twitter_igraph = ig.Graph.Read_Ncol('p-medians results\\dolphin_edge.txt', directed=True)
#print(twitter_igraph)
g = [0,1,0,1,0,1]  # known solution
c = [0,1,0,1,1,1]  # found solution

# create graph with coloring g
#graf = ig.Graph(6)

#graf.add_edges([(i,(i + 2) % 6) for i in range(6)])
#graf['colors'] = g

f = open('results_measures.txt', 'w')

average = 'micro' # 'micro', 'macro', 'samples','weighted', 'binary', None
print('NMI', normalized_mutual_info_score(g, c), file=f)
print('AMI', adjusted_mutual_info_score(g, c), file=f)
print('ARI', adjusted_rand_score(g, c), file=f)
print('phi', conductance(graf, c)[0], file=f)
print('gamma', coverage(graf, c), file=f)
print('Q', graf.modularity(c), file=f)
print('F1', f1_score(g, c, average = average), file=f)
print('F2', fbeta_score(g, c, beta=2, average = average), file=f)
print('FM', fowlkes_mallows_score(g, c), file=f)
print('JI', jaccard_score(g, c, average = average), file=f)
print('V',  v_measure_score(g, c), file=f)
print('# clusters', len(set(c)), file=f)

f.close()