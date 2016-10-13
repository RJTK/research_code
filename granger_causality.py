import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import networkx as nx

from matplotlib import pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.base.datetools import dates_from_str
from itertools import product
from memory_limiter import limit_memory_as

#X = prepare_dates(pd.DataFrame(X.T))
def prepare_dates(X):
  '''
  Changes the index of X into datetime objects.
  '''
  X.index = pd.date_range('1/1/2016', periods = len(X.index), freq = 'D')
  return X

def gc_graph(X, p = 2, signif = 0.01):
  '''
  X should be a pandas dataframe ready to be consumed by VAR(-)

  p is the model order we will use.

  We then produce a granger causality graph, where GC is tested w.r.t.
  the whole information set.
  '''
  G = nx.DiGraph()
  G.add_nodes_from(X.columns.values)
  
  model = VAR(X)
  results = model.fit(p)

  #Suppress output from test_causality
  import sys
  stdout_real = sys.stdout
  sys.stdout = open('/dev/null', 'w')

  #itertools product
  for e in product(X.columns.values, X.columns.values):
    gc = results.test_causality(*e, signif = signif)
    if gc['conclusion'] == 'reject':
      G.add_edge(e[0], e[1])

  sys.stdout = stdout_real
  return G

if __name__ == '__main__':
  import resource as r

  from data_synth import data2
  from matplotlib import pyplot as plt

  mem_lim = 7000*1e6 #Experimentally determined, no real significance
  limit_memory_as(mem_lim)

  p = 2
  signif = 0.001

  K = range(0, 31, 10)[1:]
  T = range(0, 500, 100)[1:]

  result = np.zeros((len(K), len(T)))

  np.random.seed(1)
  
  for i, k in enumerate(K):
    print 'K = %d' % k
    for j, t in enumerate(T):
      try:
        G, _, X, _ = data2(p, K = k, T = t)
        X = prepare_dates(pd.DataFrame(X.T))
        GC = gc_graph(X, p, signif = signif)

        G_A = nx.adjacency_matrix(G).toarray()
        GC_A = nx.adjacency_matrix(GC).toarray()

        errs = sum(sum(G_A != GC_A))
        result[i, j] = errs
      except MemoryError:
        print 'MemoryError at (K, T) = (%d, %d)' % (k, t)
        continue

  print result
  plt.imshow(result)
