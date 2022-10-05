import networkx as nx
import numpy as np
import math
import pandas as pd
from pulp import *
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


#minimum and maximum number of clusters
Cl_min = 2
Cl_max = 10

#iris = pd.read_csv("c:/Documents/agoston/janez/z_iris.txt",sep=' ')
wine = pd.read_csv("c:/Documents/agoston/janez/z_wine.txt",sep=' ')
#Dist_EU = squareform(pdist(iris))
Dist_EU = squareform(pdist(wine))
size=len(Dist_EU)

# array for assignments
Assign = np.zeros((size,Cl_max-Cl_min+1),dtype=int)

DistList = list(range(size*size))
for i in range(size*size):
	DistList[i] = Dist_EU[math.trunc(i/size)][i-(math.trunc((i)/size))*size]


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
c = LpAffineExpression([ (x[i],DistList[i]) for i in range(size*size)])
#c = LpAffineExpression([ (x[i],DistList[i]*DistList[i]) for i in range(size*size)])

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
    print("cluster: ",cluster_i)
    prob.constraints['cl_num'] = c == cluster_i
    filename = "KMED" + str(cluster_i) +".lp"
    prob.writeLP(filename)
#    prob.solve()
#    prob.solve(solver = GUROBI_CMD())
#    print("Cluster number:", cluster_i)
#    print("Status:", LpStatus[prob.status])
#    print("Objective function value: ", value(prob.objective))
#    print("Running time",prob.solutionTime)
