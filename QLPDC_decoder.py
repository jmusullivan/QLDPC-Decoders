"""QLPDC decoder"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import galois
from pymatching import Matching


GF = galois.GF(2)


def tanner_graph(H: np.ndarray):
    "Create tanner graph from a parity check matrix H."
    m, n = H.shape
    T = nx.Graph()

    T.H = H
    # nodes
    T.VD = [i for i in range(n)]
    T.VC = [-j-1 for j in range(m)]

    # add them, with positions
    for i, node in enumerate(T.VD):
        T.add_node(node, pos=(i-n/2, 0), label='$d_{'+str(i)+'}$')
    for j, node in enumerate(T.VC):
        T.add_node(node, pos=(j-m/2, 1), label='$c_{'+str(j)+'}$')

    # add checks to graph
    for j, check in enumerate(H):
        for i, v in enumerate(check):
            if v:
                T.add_edge(-j-1, i)

    return T

def draw_tanner_graph(T, highlight_vertices=None):
    
    "Draw the graph. highlight_vertices is a list of vertices to be colored."
    pl=nx.get_node_attributes(T,'pos')
    lbls = nx.get_node_attributes(T, 'label')

    nx.draw_networkx_nodes(T, pos=pl, nodelist=T.VD, node_shape='o')
    nx.draw_networkx_nodes(T, pos=pl, nodelist=T.VC, node_shape='s')
    nx.draw_networkx_labels(T, pos=pl, labels=lbls)

    nx.draw_networkx_edges(T, pos=pl)

    if highlight_vertices:
        nx.draw_networkx_nodes(T,
                               pos=pl,
                               nodelist=[int(v[:]) for v in highlight_vertices if v[0] == 'd'],
                               node_color='red',
                               node_shape='o')
        nx.draw_networkx_nodes(T,
                       pos=pl,
                       nodelist=[-int(v[:])-1 for v in highlight_vertices if v[0] == 'c'],
                       node_color='red',
                       node_shape='s')

    plt.axis('off');

# these four functions allow us to convert between 
# (s)tring names of vertices and (i)nteger names of vertices
def s2i(node):
    return int(node[1]) if node[0] == 'd' else -int(node[1])-1

def i2s(node):
    return f'd{node}' if node>=0 else f'c{-node-1}'

def ms2i(W: set):
    return set(map(s2i, W))

def mi2s(W: set):
    return set(map(i2s, W))



#################################################
#################################################

def interior(T, W):
    "Determine interior of vertex subset W of Tanner graph T."
    IntW = set()
    for v in W:
        if set(nx.neighbors(T,v)).issubset(W):
            IntW.add(v)
    return IntW


def solvable_system(A,b):
    "Determines if there is a solution to Ax=b."
    A_rank = np.linalg.matrix_rank(A)

    # create augmented matrix
    Ab = np.hstack((A, np.atleast_2d(b).T))

    # Must be true for solutions to be consistent
    return A_rank == np.linalg.matrix_rank(Ab)

def solve_underdetermined_system(A, b):
    "Returns a random solution to Ax=b."
    n_vars = A.shape[1]
    A_rank = np.linalg.matrix_rank(A)

    # create augmented matrix
    Ab = np.hstack((A, np.atleast_2d(b).T))

    # Must be true for solutions to be consistent
    if A_rank != np.linalg.matrix_rank(Ab):
        raise Exception("No solution exists.")

    # reduce the system
    Abrr = Ab.row_reduce()

    # additionally need form in which the identity
    # is moved all the way to the left. Do some
    # column swaps to achieve this.
    swaps = []
    for i in range(min(Abrr.shape)):
        if Abrr[i,i] == 0:
            for j in range(i+1,n_vars):
                if Abrr[i,j] == 1:
                    Abrr[:, [i,j]] = Abrr[:, [j,i]]
                    swaps.append((i,j))
                    break

    # randomly generate some independent variables
    n_ind = n_vars - A_rank
    ind_vars = GF(np.zeros(n_ind,dtype = int))

    # compute dependent variables using reduced system and dep vars
    dep_vars = -Abrr[:A_rank,A_rank:n_vars]@ind_vars + Abrr[:A_rank,-1]

    # x is just concatenation of the two
    x = np.hstack((dep_vars, ind_vars))

    # swap the entries of x according to the column swaps earlier
    # to get the solution to the original system.
    for s in reversed(swaps):
        x[s[0]], x[s[1]] = x[s[1]], x[s[0]]

    return x


def is_valid_cluster(T, syndrome, cluster):
    "Given a syndrome and cluster, determines if is is valid."

    cluster_interior = interior(T, cluster)

    data_qubits_in_interior = sorted([i for i in cluster_interior if i>=0])
    check_qubits_in_cluster = sorted([-i-1 for i in cluster if i<0])

    GF = galois.GF(2)
    A = GF(T.H[check_qubits_in_cluster][:,data_qubits_in_interior])
    b = GF([syndrome[i] for i in check_qubits_in_cluster])

    solved = solvable_system(A,b)

    return solved


def find_valid_correction(T, syndrome, cluster):

    cluster_interior = interior(T, cluster)

    data_qubits_in_interior = sorted([i for i in cluster_interior if i>=0])
    check_qubits_in_cluster = sorted([-i-1 for i in cluster if i<0])

    GF = galois.GF(2)
    A = GF(T.H[check_qubits_in_cluster][:,data_qubits_in_interior])
    b = GF([syndrome[i] for i in check_qubits_in_cluster]) 

    sol = solve_underdetermined_system(A,b)

    return sol, data_qubits_in_interior


def cluster_neighbors(T, cluster):
    nbrs = set()
    for v in cluster:
        nbrs.update(set(nx.neighbors(T,v)))
    return nbrs

#################################################
#################################################

#Actual decoder

def my_ldpc_decoder(T, syndrome):

    n = len(T.VD)

    # assign each syndrome to its own cluster
    K = [set([-i-1]) for i,s in enumerate(syndrome) if s]

    valid_clusters = lambda K : np.array([is_valid_cluster(T, syndrome, Ki) for Ki in K]).all()
    n_finite_loop = 0
    # grow clusters till all are valid
    while True:
        # if during the loop below some clusters are
        # merged, then we restart the loop
        K_updated = False
        n_finite_loop+=1
        # When clusters are merged, then later need to
        # delete the one that was added to the other.
        to_delete = set()
        #print(f"Restarting {K}")
        for i, Ki in enumerate(K):
            # print(f'K{i} = {Ki}')
            if not is_valid_cluster(T, syndrome, Ki):
                nbrs = cluster_neighbors(T, Ki)
                Ki.update(nbrs)
                # print(f'K{i} with nbrs = {Ki}')
                # for loop to check if clusters need to
                # be merged with this newly grown one
                for j, Kj in enumerate(K):
                    if j < i and (not nbrs.isdisjoint(Kj)) and (not is_valid_cluster(T, syndrome, Kj)):
                        Kj.update(Ki)
                        # print(f'K{j} with K{i} = {Kj}')
                        to_delete.update([i])
                        K_updated = True
                    elif j > i and (not nbrs.isdisjoint(Kj)) and (not is_valid_cluster(T, syndrome, Kj)):
                        Ki.update(Kj)
                        # print(f'K{i} with K{j} = {Ki}')
                        to_delete.update([j])
                        K_updated = True
                if K_updated:
                    break
        for i in reversed(sorted(to_delete)):
            K.pop(i)
        if valid_clusters(K):
            break
        if n_finite_loop>100:
            print('loop failed to terminate')
            break

    # determine error estimate using valid clusters
    e_estimate = np.array([0]*n, dtype=int)
    for i, Ki in enumerate(K):
        correction, correction_data_qubits = find_valid_correction(T, syndrome, Ki)
        e_estimate[correction_data_qubits] = correction

    return e_estimate, K 

