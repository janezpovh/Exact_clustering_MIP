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
Assign = np.zeros((size,Cl_max-Cl_min+1))

DistList = list(range(size*size))
for i in range(size*size):
	DistList[i] = Dist_EU[math.trunc(i/size)][i-(math.trunc((i)/size))*size]

prob = LpProblem("MSSCluster_Problem")


#arrays of decision variable
z_name = [str(i) for i in list(range(size*size))]
zeta_name = [str(i) for i in list(range(size*size))]

# filling list z and zeta with real variable names
for row_i in range(size):
	for column_i in range(size):
		z_name[row_i*size+column_i] = "z"+str(row_i+1)+"_"+str(column_i+1)
		zeta_name[row_i*size+column_i] = "zeta"+str(row_i+1)+"_"+str(column_i+1)

z = [LpVariable(z_name[i], lowBound = 0, upBound = 1) for i in range(size*size)]
zeta = [LpVariable(zeta_name[i], lowBound = 0, upBound = 1, cat = 'Binary') for i in range(size*size)]

#objective function
#c = LpAffineExpression([ (z[i],DistList[i]) for i in range(size*size)])
c = LpAffineExpression([ (z[i],DistList[i]*DistList[i]) for i in range(size*size)])

prob += c

# constraint: sum of z variables = 1
for row_i in range(size):
	c = LpAffineExpression([ (z[i],1) for i in range(row_i*size,(row_i+1)*size)])
	constr_name = "Assign"+str(row_i)
	prob += c == 1, constr_name

# constraints: with zi_j and zj_i
const_count = 0
for row_i in range(size):
	for column_i in range(size):
		if row_i < column_i:
			c = LpAffineExpression([ (z[row_i*size+column_i],1),(z[column_i*size+row_i],-1)])
			constr_name = "Symmet"+str(row_i)+"_"+str(column_i)
			prob += c == 0, constr_name
		if row_i != column_i:
			constr_name = "zeta_less"+str(row_i)+"_"+str(column_i)
			c = LpAffineExpression([ (zeta[row_i*size+column_i],1),(z[row_i*size+column_i],-1)])
			prob += c >= 0, constr_name
			constr_name = "z_less"+str(row_i)+"_"+str(column_i)
			c = LpAffineExpression([ (z[row_i*size+row_i],-1),(z[row_i*size+column_i],1)])
			prob += c <= 0, constr_name
			constr_name = "cutA"+str(const_count)
			c = LpAffineExpression([ (zeta[row_i*size+column_i],1),(z[row_i*size+column_i],-1*(size-2+1))])
			prob += c <= 0, constr_name
			constr_name = "cutB"+str(const_count)
			c = LpAffineExpression([ (zeta[row_i*size+column_i],1),(z[row_i*size+row_i],1*(size-2+1)),(z[row_i*size+column_i],-1*(size-2+1))])
			prob += c >= 1, constr_name
			constr_name = "cutC"+str(const_count)
			c = LpAffineExpression([ (zeta[row_i*size+column_i],1),(z[row_i*size+row_i],1),(z[row_i*size+column_i],-1)])
			prob += c <= 1, constr_name

			const_count = const_count + 1

# constraints: triangle inequalities
for i in range(size):
	for j in range(size):
		for k in range(size):
			if i!=j and i!=k and j<k:
				c = LpAffineExpression([ (z[i*size+i],-1),(z[i*size+j],1),(z[i*size+k],1),(z[j*size+k],-1)])
				constr_name = "triang"+str(i)+"_"+str(j)+"_"+str(k)
				prob += c <= 0, constr_name

# constraints: cluster number
c = LpAffineExpression([(z[i*size+i],1) for i in range(size)])
prob += c == 2, 'cl_num'


for cluster_i in range(Cl_min,Cl_max+1):
    print("Cluster number:", cluster_i)
    prob.constraints['cl_num'] = c == cluster_i
    const_count = 0
    for row_i in range(size):
    	for column_i in range(size):
    		if row_i != column_i:
    			constr_name = "cutA"+str(const_count)
    			prob.constraints[constr_name][z[row_i*size+column_i]] =  -1*(size-cluster_i+1)
    			constr_name = "cutB"+str(const_count)
    			prob.constraints[constr_name][z[row_i*size+column_i]] =  -1*(size-cluster_i+1)
    			prob.constraints[constr_name][z[row_i*size+row_i]] =  1*(size-cluster_i+1)
    			const_count = const_count + 1

    filename = "MSS" + str(cluster_i) +".lp"
    prob.writeLP(filename)
#    prob.solve()
#    prob.solve(solver = GUROBI_CMD(timeLimit=10000))
#    print("Status:", LpStatus[prob.status])
#    print("Objective function value: ", value(prob.objective))
#    print("Running time",prob.solutionTime)

