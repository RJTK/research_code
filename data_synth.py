'''
Produce a bunch of synthetic time series data
'''

import math
import numpy as np
import networkx as nx

from matplotlib import pyplot as plt

from scipy import signal
from scipy import stats

from util import plot_filter_pz, plot_filter, plot_filter_PSD,\
  random_arma, power_transfer

#----------WHITE NOISE GENERATION---------
def iid_bernoulli(p, n):
  '''
  An iid sampling sequence.  Each element of the sequence is iid Ber(p).  e.g.
  each element is 1 with probability p and 0 with probability 1 - p.
  '''
  return stats.bernoulli.rvs(p, size = n)

def iid_normal(size = 1, mu = 0, s2 = 1):
  '''
  An iid sequence of N(mu, s2) random data.  This is white noise.
  '''
  return stats.norm.rvs(loc = mu, scale = math.sqrt(s2), size = size)

#----------RANDOM VAR GENERATOR---------
def random_var(n, p, p_radius = 0.9, G_A = None):
  '''
  Generates n by n random AR(p) filters and distributes their
  coefficients appropriately amongst p n by n matrices.

  G_A should be the adjacency matrix of a graph used to specify
  the topology.

  ***
  This however, does NOT produce a stable system.  The graph's
  topology affects the stability.
  ***
  '''
  if G_A == None:
    G_A = np.ones((n, n))
  else:
    assert G_A.shape[0] == G_A.shape[1], 'G_A must be square'
    assert G_A.shape[0] == n, 'Dimension n must match that of G_A'
  A = np.zeros((p, n, n))
  for i in range(n):
    for j in range(n):
      if G_A[i, j]:
        b, a = random_arma(p, 1, p_radius = p_radius)
        A[:, i, j] = -a[1:]
  return A

#I think it's totally unnecessary to actually generate a literal graph
#I am likely much better off generating random VARMA models...
#----------GRAPH GENERATORS---------
def ER_directed(n, p = 0.5):
  '''
  Generates the adjacency matrix of a directed erdos renyi G(K, q) graph.
  
  Edges appear with probability p.  Graph is size n by n
  '''
  G = stats.bernoulli.rvs(p, size = (n, n))
  return G

def gn_graph(n, kernel = None, seed = 0):
  '''
  Creates and returns a growing network grpah from networkx.  We return
  it's adjacency matrix, which is stored as a sparse matrix.

  Creating an ARMA model from this graph will infact lead to a stable
  system (as long as each individual filter is stable) since we produce
  a tree shaped graph.
  '''
  G = nx.gn_graph(n, kernel = kernel, seed = seed)
  return G

def gcgraph_gn(n, p, q, seed = 0):
  '''
  INCOMPLETE

  This function creates a random growing network graph using
  nx.gn_graph with n nodes and the given seed.  It then generates a
  random ARMA(p, q) filter for each edge using the random_arma
  function from util.py.  We then do a topological sort of the
  graph and put unit variance white noise on the driver nodes.
  Finally we associate with each edge the spectrum of a process.
  
  Of course there are a great many paramters we could modify with this
  method.  This is just a first pass to see if I can get something to
  work at all.
  '''
  #Generate a graph with a filter on each edge
  G = nx.gn_graph(n, seed = seed)
  for e in G.edges_iter():
    H = random_arma(p, q)
    G.add_edge(*e, filter_ba = H)

  #Identify the driving nodes and a topological ordering
  #We remove the driving nodes from the topological order
  topo_order = nx.topological_sort(G)
  driving_nodes = set()
  for v in G.nodes_iter():
    if len(G.predecessors(v)) == 0:
      driving_nodes.add(v)
  for i, v in enumerate(topo_order):
    if v in driving_nodes:
      del topo_order[i]

  #Assign a spectrum to the driving nodes (use white noise)
  w = np.linspace(0, np.pi, 1000) #The frequency points to use
  P_white = np.ones_like(w)
  
  for v in driving_nodes:
    G.add_node(v, PSD = P_white)

  return G

#---------DATA SYNTHESIZERS----------
def data1(p = 2, K = 3, q = 0.6, T = 200, s2 = 1):
  '''
  Random graph: ER_directed(K, q)
  '''
  np.random.seed(2)

  G_A = ER_directed(K, q)
  A = random_var(K, p, G_A = G_A)
  #A = A*G
  e0 = iid_normal(size = (K, 1))
  X = np.hstack((np.zeros((K, p - 1)), e0))

  for t in range(T - 1):
    et = iid_normal(size = (K, 1))
    xt = sum(np.dot(A[i, :, :], X[:, -(i + 1)])
             for i in range(p)).reshape(K, 1) + et
    X = np.hstack((X, xt))
  X = X[:, p - 1 : ]
  t = np.arange(0, T)
  return G_A, t, X, A

def data2(p = 2, K = 3, T = 200, s2 = 1, seed = None):
  '''
  Random graph: gn_graph_adj.
  '''
  if seed:
    np.random.seed(seed)
  
  G = gn_graph(K)
  G_A = nx.adjacency_matrix(G)
  A = random_var(K, p, G_A = G_A)
  e0 = iid_normal(size = (K, 1))
  X = np.hstack((np.zeros((K, p - 1)), e0))
  
  for t in range(T - 1):
    et = iid_normal(size = (K, 1))
    xt = sum(np.dot(A[i, :, :], X[:, -(i + 1)])
             for i in range(p)).reshape(K, 1) + et
    X = np.hstack((X, xt))
  X = X[:, p - 1 : ]
  t = np.arange(0, T)
  return G, t, X, A

#---------TESTS----------
def test1():
  '''
  Tests the generation of random arma filters and plots their responses
  '''
  b, a = random_arma(9, 5, p_radius = 0.95, z_radius = 1.5, k = 1)
  plot_filter(b, a, plot_gd = True)
  plot_filter_pz(b, a)
  plot_filter_PSD(b, a)
  plt.show()
  return

def test2():
  '''
  Tests power transfer
  '''
  b1, a1 = random_arma(9, 5, p_radius = 0.95, z_radius = 1.5, k = 1)
  b2, a2 = random_arma(9, 5, p_radius = 0.95, z_radius = 1.5, k = 1)
  plot_filter(b1, a1)
  plot_filter(b2, a2)
  plt.show()

  s2 = 1
  w = np.linspace(0, np.pi, 1000)
  P_in = s2*np.ones_like(w)
  P1 = power_transfer(w, P_in, (b1, a1))
  P2 = power_transfer(w, P1, (b2, a2))

  plt.plot(w, 20*np.log10(P_in), linewidth = 2, label = '$P_{in}$')
  plt.plot(w, 20*np.log10(P1), linewidth = 2, label = '$P_1$')
  plt.plot(w, 20*np.log10(P2), linewidth = 2, label = '$P_2$')

  plt.title('PSD')
  plt.ylabel('$P(e^{jw})$')
  plt.xlabel('$w$ [rad/s]')
  plt.legend()
  plt.show()
  return

if __name__ == '__main__':
#  test1()
  test2()
  exit(0)
  N = 100000
  x = iid_normal(N)
  t, y = signal.dlsim((b, a, 1), x)
  y = y.T[0]
  f, P = signal.welch(y)
  plt.plot(f, P)
  plt.show()
  raw_input('continue?')

#  G = nx.DiGraph()
#  G.add_nodes_from(range(5))
#  G.add_edge(0, 1, a = [1], b = [1, -1])
