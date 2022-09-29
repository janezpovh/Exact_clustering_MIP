import networkx as nx
import numpy as np
import math
import pandas as pd
from pulp import *
from Small_kmed import *


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
#GFC = nx.read_gml("c:/Users/Kolos/Janez/polbooks.gml")
#GFC = nx.read_gml("c:/Documents/agoston/janez/polbooks.gml")
#GFC = nx.read_gml("c:/Documents/agoston/janez/dolphins.gml")
#GFC = nx.read_gml("c:/Documents/agoston/janez/football.gml")
GFC = nx.read_gml("c:/Documents/agoston/janez/karate.gml")
#GFC = nx.read_gml("c:/Documents/agoston/janez/UKfaculty.gml")

Nodes = list(GFC.nodes)

size = len(Nodes)

# array for assignments
Assign = np.zeros((size,Cl_max-Cl_min+1))

GeodDist = np.zeros((size,size))

for edge_from in range(size):
	for edge_to in range(size):
		GeodDist[edge_from][edge_to] = len(nx.shortest_path(GFC,Nodes[edge_from],Nodes[edge_to]))-1

GeodDistList = list(range(size*size))
for i in range(size*size):
	GeodDistList[i] = GeodDist[math.trunc(i/size)][i-(math.trunc((i)/size))*size]


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
#c = LpAffineExpression([ (z[i],GeodDistList[i]) for i in range(size*size)])
c = LpAffineExpression([ (z[i],GeodDistList[i]*GeodDistList[i]) for i in range(size*size)])

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

prob.writeLP("MSS.lp")

print("cluster i start")

for cluster_i in range(Cl_min,Cl_max+1):
    prob.constraints['cl_num'] = c == cluster_i
    const_count = 0
    for row_i in range(size):
    	for column_i in range(size):
    		if row_i != column_i:
    			constr_name = "cutA"+str(const_count)
#    			print(constr_name)
#    			print(z_name[row_i*size+column_i])
    			prob.constraints[constr_name][z[row_i*size+column_i]] =  -1*(size-cluster_i+1)
    			constr_name = "cutB"+str(const_count)
    			prob.constraints[constr_name][z[row_i*size+column_i]] =  -1*(size-cluster_i+1)
    			prob.constraints[constr_name][z[row_i*size+row_i]] =  1*(size-cluster_i+1)
    			const_count = const_count + 1
    prob.writeLP("MSS.lp")
#    prob.solve()
    prob.solve(solver = GUROBI_CMD(timeLimit=10000))
    print("Cluster number:", cluster_i)
    print("Status:", LpStatus[prob.status])
    if (LpStatus[prob.status] == "Optimal"):
        print("Chekk is OK")
    print("Objective function value: ", value(prob.objective))
    print("Running time",prob.solutionTime)
    if (LpStatus[prob.status] == "Optimal"):
# Optimal solution printing calculation block start
        print("Chekk is OK")
        Obj_Val[cluster_i-Cl_min] = value(prob.objective)
        Run_Time[cluster_i-Cl_min] = prob.solutionTime

        Cluster_index = np.zeros((size),dtype=int)
        Assignment = np.zeros((size),dtype=int)
        Centers = np.zeros((size),dtype=int)    
        Cluster_sizes = np.zeros((cluster_i),dtype=int)
        cl_ind = 0

        for i in range(size*size):
#         print(z_name[i],value(z[i]))
            if value(z[i]) > (1/(size+1)):
                Assign[math.trunc(i/size)][cluster_i-Cl_min] = i-math.trunc(i/size)*size+1
                Assignment[math.trunc(i/size)] = i-math.trunc(i/size)*size
    
        print(Assignment)

        index_cluster_repr = np.unique(Assignment)
    
#    print(index_cluster_repr)

        cl_ind = 0
        for i in range(size):
            if Assignment[i] == i:
                Cluster_index[i] = cl_ind
                cl_ind = cl_ind + 1

        for i in range(size):
            Cluster_sizes[Cluster_index[Assignment[i]]] = Cluster_sizes[Cluster_index[Assignment[i]]] + 1
        
        


#    print(Cluster_index)

#            Cluster_sizes[Cluster_index[i-math.trunc(i/size)*size]] = Cluster_sizes[Cluster_index[i-math.trunc(i/size)*size]] + 1

        print("Cluster sizes: ",Cluster_sizes)

        Avg_Dist_WC = np.zeros((cluster_i))
        for i in range(size):
#        Avg_Dist_FC = Avg_Dist_FC + GeodDist[i,Assign[i][cluster_i-Cl_min]-1]
            if (i < size):
                for j in range(i+1,size):
                    if (Assignment[i] == Assignment[j]):
                        Avg_Dist_WC[Cluster_index[Assignment[i]]] = Avg_Dist_WC[Cluster_index[Assignment[i]]] + GeodDist[i][j]
                    
        print("Avg. Dist. WC (array) ",Avg_Dist_WC)
            
 #   Avg_Dist_FC = Avg_Dist_FC / size
 #   Avg_Dist_from_Centers[cluster_i-Cl_min] = Avg_Dist_FC
    
#    print(Avg_Dist_FC)

        Avg_Dist_WC_F = 0

        for cl_i in range(cluster_i):
            Avg_Dist_WC_F = Avg_Dist_WC_F + Avg_Dist_WC[cl_i] / Cluster_sizes[cl_i]

        Sum_Dist_within_Clusters[cluster_i-Cl_min] = Avg_Dist_WC_F

        print("Sum. Dist. WC ",Avg_Dist_WC_F)

        Avg_Dist_WC_F = 0

        for cl_i in range(cluster_i):
            if (Cluster_sizes[cl_i] >= 2):
                Avg_Dist_WC_F = Avg_Dist_WC_F + (Cluster_sizes[cl_i]/size) * (Avg_Dist_WC[cl_i] /( 0.5 * Cluster_sizes[cl_i]*(Cluster_sizes[cl_i]-1)))

        Avg_Dist_within_Clusters[cluster_i-Cl_min] = Avg_Dist_WC_F

        print("Avg. Dist. WC ",Avg_Dist_WC_F)
    
        for cl_i in range(cluster_i):
            GeodDistSmall = np.zeros((Cluster_sizes[cl_i],Cluster_sizes[cl_i]))
            Kmed_small_ind = [index for index,value in enumerate(Assignment) if value == index_cluster_repr[cl_i]]
#        print(Kmed_small_ind)
            for i in range(Cluster_sizes[cl_i]):
                for j in range(Cluster_sizes[cl_i]):
#                print(cl_i," ",i," ",j," ",Cluster_sizes[cl_i]," ",Kmed_small_ind[i]," ",Kmed_small_ind[j])
                    GeodDistSmall[i][j] = GeodDist[Kmed_small_ind[i]][Kmed_small_ind[j]]
        
#        print(small_kmed(GeodDistSmall))
            cent_act = Kmed_small_ind[small_kmed(GeodDistSmall)]
    
            for i in range(Cluster_sizes[cl_i]):
                Centers[Kmed_small_ind[i]] = cent_act
    
#    print(Centers)
    
        Avg_Dist_FC = 0
        for i in range(size):
            Avg_Dist_FC = Avg_Dist_FC + GeodDist[i][Centers[i]]

        Avg_Dist_FC = Avg_Dist_FC / size
        Avg_Dist_from_Centers[cluster_i-Cl_min] = Avg_Dist_FC
    
        print("Avg. Dist. FC ",Avg_Dist_FC)

# Optimal solution printing calculation block end

    else:
        print("No optimal solution")
        
np.savetxt('Assign.txt',Assign)

np.savetxt('Cluster_info.txt',np.vstack((Obj_Val,Run_Time,Avg_Dist_from_Centers,Avg_Dist_within_Clusters,Sum_Dist_within_Clusters)).T)


