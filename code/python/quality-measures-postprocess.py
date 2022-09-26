# written by B Gabrovsek
# uppdated by J Povh 2022 09 23

import igraph as ig
from sklearn.metrics import *
import numpy as np
import pandas as pd
from math import sqrt
import os
os.chdir('C:\\Users\\janez\\Documents\\MATLAB\\jpcode\\exact_clustering_MIP')




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


path_c = "results\\Optimum_clusterings"
path_g="data\\networks\\"

# data political books
optim_clustering="polBooks_exact_clustering.txt"
exact_clustering="polBooks_k_3.txt"
data_set="Political books.txt"

# data karate
optim_clustering="karate_exact_clustering.txt"
exact_clustering="karate_k_2.txt"
data_set="karate.txt"

# load graph
adj_dataframe =pd.read_csv(path_g+"\\"+data_set, delimiter=",", header=None)  # beri kot panda dataframe
adj_dataframe =np.loadtxt(path_g+"\\"+data_set)  # beri kot panda dataframe
print(adj_dataframe)
#n = int(sqrt(adj_dataframe.shape[1][1]))  # določi število vozlišč
n = int(adj_dataframe.shape[1])  # določi število vozlišč

#adj_data = adj_dataframe.to_numpy().reshape((n, n))  # pretvori v numpy tabelo
adj_data = adj_dataframe
print(adj_data)
graph_1 = ig.Graph.Adjacency(adj_data.tolist())  # pretvori v graf
# load ground truth clustering
exact_coloring = np.loadtxt(path_c+"\\"+optim_clustering, skiprows=1)[:,1] # ignore 1st column
# load experiments
data = np.loadtxt(path_c+"\\"+exact_clustering, skiprows=1)


#print(g)

# load ground truth 
#exact_df = pd.read_csv("polBooks_exact_clustering.txt", delimiter=" ")
#exact_coloring = exact_df['cluster'].to_numpy() 
# bolje:

print("exact coloring", exact_coloring[:10],"...", "size", exact_coloring.shape)




coloring_Kmed = data[:,0]	
coloring_MSS = data[:,1]
coloring_Kmed_sq = data[:,2]
coloring_MSS_sq = data[:,3]

print("Kmed", coloring_Kmed[:10], "...", "size:  ", coloring_Kmed.shape)
print("MSS", coloring_MSS[:10], "...","size:  ", coloring_MSS.shape)
print("Kmed_sq", coloring_Kmed_sq[:10], "...","size:  ", coloring_Kmed_sq.shape)
print("MSS_sq", coloring_MSS_sq[:10], "...","size:  ", coloring_MSS_sq.shape)


print()

# koda od prej

g = exact_coloring  # real solution
# create graph with coloring g
graf = graph_1
graf['colors'] = g

f = open('results\\results_measures.txt', 'a')
print('dataset   n   k   method   NMI   AMI   ARI   phi   gamma   Q   F1   F2   FM   JI   V',file=f)

methods=["Kmed","MSS","Kmed_Sq","MSS_sq"]
ind=0
for c in [coloring_Kmed, coloring_MSS, coloring_Kmed_sq, coloring_MSS_sq]:

    print("---")

    average = 'micro' # 'micro', 'macro', 'samples','weighted', 'binary', None
    print('SIZE of g',)
    NMI= normalized_mutual_info_score(g, c)
    AMI=adjusted_mutual_info_score(g, c)
    ARI=adjusted_rand_score(g, c)
    phi=conductance(graf, c)[0]
    gamma=coverage(graf, c)
    Q=graf.modularity(c)
    F1=f1_score(g, c, average = average)
    F2=fbeta_score(g, c, beta=2, average = average)
    FM=fowlkes_mallows_score(g, c)
    JI=jaccard_score(g, c, average = average)
    V=v_measure_score(g, c)
    clusters=len(set(c))
    num_nodes=len(g)
    print(data_set,num_nodes,clusters,methods[ind],NMI,AMI,ARI,phi,gamma,Q,F1,F2,FM,JI,V, file=f)
    ind=ind+1
    print('AMI', adjusted_mutual_info_score(g, c))
    print('ARI', adjusted_rand_score(g, c))
    print('phi', conductance(graf, c)[0])
    print('gamma', coverage(graf, c))
    print('Q', graf.modularity(c))
    print('F1', f1_score(g, c, average = average))
    print('F2', fbeta_score(g, c, beta=2, average = average))
    print('FM', fowlkes_mallows_score(g, c))
    print('JI', jaccard_score(g, c, average = average))
    print('V',  v_measure_score(g, c))
    print('# clusters', len(set(c)))
