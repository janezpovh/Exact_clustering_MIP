def small_kmed(GeodDist):

	import math
	import pulp
	from datetime import datetime

	size = len(GeodDist)

	GeodDistList = list(range(size*size))
	for i in range(size*size):
		GeodDistList[i] = GeodDist[math.trunc(i/size)][i-(math.trunc((i)/size))*size]

	kmed = pulp.LpProblem("K_median_Cluster_Problem")

	#arrays of decision variable
	x_name = [str(i) for i in list(range(size*size))]
	y_name = [str(i) for i in list(range(size))]

	# filling list x with real variable names
	for row_i in range(size):
		y_name[row_i] = "y"+str(row_i+1)
		for column_i in range(size):
			x_name[row_i*size+column_i] = "x"+str(row_i+1)+"_"+str(column_i+1)

	x = [pulp.LpVariable(x_name[i], lowBound = 0, upBound = 1, cat = 'Binary') 	for i in range(size*size)]
	y = [pulp.LpVariable(y_name[i], lowBound = 0, upBound = 1, cat = 'Binary') for i in range(size)]

	#objective function
	c = pulp.LpAffineExpression([ (x[i],GeodDistList[i]) for i in range(size*size)])

	kmed += c

# constraints: every element can be assigned to exactly one cluster center
	for row_i in range(size):
		c = pulp.LpAffineExpression([ (x[i],1) for i in range(row_i*size,(row_i+1)*size)])
		kmed += c == 1

	# constraints: xi_j <= yj
	for row_i in range(size):
		for column_i in range(size):
			c = pulp.LpAffineExpression([ (x[row_i*size+column_i],1),(y[column_i],-1)])
			kmed += c <= 0

	# constraints: cluster number
	c = pulp.LpAffineExpression([(y[i],1) for i in range(size)])
	kmed += c == 1, 'cl_num'
	
	now = datetime.now()
	nev = "Kmed1" + now.strftime("%H%M%S%f") + ".lp"
#	kmed.writeLP(nev)

	#    kmed.solve()
	kmed.solve(solver = pulp.GUROBI_CMD())

	center = -1
	for i in range(size):
		if (pulp.value(y[i]) == 1):
			center = i
	

	return center
