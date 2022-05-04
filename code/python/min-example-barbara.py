import igraph as ig




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


g = ig.Graph(4)
g.add_edges([(0, 2),(0, 3),(1,2),(1,3)])

print("conductance", conductance(g, [0,0,0,1]))
print("coverage", coverage(g, [0,0,0,1]))
