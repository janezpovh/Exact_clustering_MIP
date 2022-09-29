import networkx as nx
import numpy as np
import math
import pandas as pd
from pulp import *

#minimum and maximum number of clusters
Cl_min = 2
Cl_max = 10


# array for objective values (distnce minimum)
Obj_Val = np.zeros((Cl_max-Cl_min+1))
Run_Time = np.zeros((Cl_max-Cl_min+1))
Avg_Dist_from_Centers = np.zeros((Cl_max-Cl_min+1))
Sum_Dist_within_Clusters = np.zeros((Cl_max-Cl_min+1))
Avg_Dist_within_Clusters = np.zeros((Cl_max-Cl_min+1))


# GFC = GraphForClustering
#GFC = nx.read_gml("c:/Users/Kolos/Janez/karate.gml")
#GFC = nx.read_gml("c:/Documents/agoston/janez/polbooks.gml")
GFC = nx.read_gml("c:/Documents/agoston/janez/dolphins.gml")
#GFC = nx.read_gml("c:/Documents/agoston/janez/football.gml")
#GFC = nx.read_gml("c:/Documents/agoston/janez/karate.gml")

Nodes = list(GFC.nodes)

size = len(Nodes)

# array for assignments
Assign = np.zeros((size,Cl_max-Cl_min+1),dtype=int)

GeodDist = np.zeros((size,size))

for edge_from in range(size):
	for edge_to in range(size):
		GeodDist[edge_from][edge_to] = len(nx.shortest_path(GFC,Nodes[edge_from],Nodes[edge_to]))-1

GeodDistList = list(range(size*size))
for i in range(size*size):
	GeodDistList[i] = GeodDist[math.trunc(i/size)][i-(math.trunc((i)/size))*size]


prob = LpProblem("K_median_Cluster_Problem")


#arrays of decision variable
x_name = [str(i) for i in list(range(size*size))]
y_name = [str(i) for i in list(range(size))]

# filling list x with real variable names
for row_i in range(size):
	y_name[row_i] = "y"+str(row_i+1)
	for column_i in range(size):
		x_name[row_i*size+column_i] = "x"+str(row_i+1)+"_"+str(column_i+1)

x = [LpVariable(x_name[i], lowBound = 0, upBound = 1, cat = 'Binary') for i in range(size*size)]
y = [LpVariable(y_name[i], lowBound = 0, upBound = 1, cat = 'Binary') for i in range(size)]

#objective function
c = LpAffineExpression([ (x[i],GeodDistList[i]) for i in range(size*size)])
c = LpAffineExpression([ (x[i],GeodDistList[i]*GeodDistList[i]) for i in range(size*size)])

prob += c

# constraints: every element can be assigned to exactly one cluster center
for row_i in range(size):
	c = LpAffineExpression([ (x[i],1) for i in range(row_i*size,(row_i+1)*size)])
	prob += c == 1

# constraints: xi_j <= yj
for row_i in range(size):
	for column_i in range(size):
		c = LpAffineExpression([ (x[row_i*size+column_i],1),(y[column_i],-1)])
		prob += c <= 0

# constraints: cluster number
c = LpAffineExpression([(y[i],1) for i in range(size)])
prob += c == 1, 'cl_num'

for cluster_i in range(Cl_min,Cl_max+1):
    prob.constraints['cl_num'] = c == cluster_i
    prob.writeLP("KMED.lp")
#    prob.solve()
    prob.solve(solver = GUROBI_CMD())
    print("Cluster number:", cluster_i)
    print("Status:", LpStatus[prob.status])
    print("Objective function value: ", value(prob.objective))
    print("Running time",prob.solutionTime)
    Obj_Val[cluster_i-Cl_min] = value(prob.objective)
    Run_Time[cluster_i-Cl_min] = prob.solutionTime

    Cluster_index = np.zeros((size),dtype=int)
    Cluster_sizes = np.zeros((cluster_i),dtype=int)
    cl_ind = 0
    for i in range(size):
        if value(y[i]) == 1:
            Cluster_index[i] = cl_ind
            cl_ind = cl_ind + 1
    

    for i in range(size*size):
         if value(x[i]) == 1:
             Assign[math.trunc(i/size)][cluster_i-Cl_min] = i-math.trunc(i/size)*size+1
             Cluster_sizes[Cluster_index[i-math.trunc(i/size)*size]] = Cluster_sizes[Cluster_index[i-math.trunc(i/size)*size]] + 1

    print(Cluster_sizes)

    Avg_Dist_FC = 0
    Avg_Dist_WC = np.zeros((cluster_i))
    for i in range(size):
        Avg_Dist_FC = Avg_Dist_FC + GeodDist[i,Assign[i][cluster_i-Cl_min]-1]
        if (i < size):
            for j in range(i+1,size):
                if (Assign[i][cluster_i-Cl_min] == Assign[j][cluster_i-Cl_min]):
                    Avg_Dist_WC[Cluster_index[Assign[i][cluster_i-Cl_min]-1]] = Avg_Dist_WC[Cluster_index[Assign[i][cluster_i-Cl_min]-1]] + GeodDist[i][j]
                    
    print(Avg_Dist_WC)
    print(Avg_Dist_FC)
            
    Avg_Dist_FC = Avg_Dist_FC / size
    Avg_Dist_from_Centers[cluster_i-Cl_min] = Avg_Dist_FC
    
    print(Avg_Dist_FC)

    Avg_Dist_WC_F = 0

    for cl_i in range(cluster_i):
        Avg_Dist_WC_F = Avg_Dist_WC_F + Avg_Dist_WC[cl_i] / Cluster_sizes[cl_i]

    Sum_Dist_within_Clusters[cluster_i-Cl_min] = Avg_Dist_WC_F

    print(Avg_Dist_WC_F)

    Avg_Dist_WC_F = 0

    for cl_i in range(cluster_i):
        if (Cluster_sizes[cl_i] >= 2):
            Avg_Dist_WC_F = Avg_Dist_WC_F + (Cluster_sizes[cl_i]/size) * (Avg_Dist_WC[cl_i] /( 0.5 * Cluster_sizes[cl_i]*(Cluster_sizes[cl_i]-1)))

    Avg_Dist_within_Clusters[cluster_i-Cl_min] = Avg_Dist_WC_F

    print(Avg_Dist_WC_F)



np.savetxt('Assign.txt',Assign)

np.savetxt('Cluster_info.txt',np.vstack((Obj_Val,Run_Time,Avg_Dist_from_Centers,Avg_Dist_within_Clusters,Sum_Dist_within_Clusters)).T)

#np.savetxt('GeodDist.txt',GeodDist)
