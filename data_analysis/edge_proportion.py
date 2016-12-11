'''Script for estimating edge proportions'''

import sys

import pandas as pd
import numpy as np
import spams as spm

from multiprocessing import Pool
from scipy.sparse import bsr_matrix
from random import shuffle
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib import patches
from matplotlib.widgets import Slider
from progressbar import Bar, Percentage, ETA, ProgressBar, SimpleProgress

#My own stuff
from granger_models.resource_limiter import limit_memory_as
from granger_models.data_manipulation import *
from granger_models.cross_validation import *
from granger_models.ts_models import *
from granger_models.graph_estimator import *
from filtering.data_synth import data_iidG_ER

DATA_DIR = '/home/ryan/Documents/academics/research/' \
           'granger_causality/software/datasets/synthetic/'

num_threads = 2
spams_threads = 4

#simulation parameters
file_prefix = 'iidG_ER_'
n_min = 50
n_max = 50
c = 25 #samples per possible edge
r = 0.65
q = 0.5
s2 = .1
p = 2
model = spams_lasso
delta = 1e-6
K = 5

#These functions use the above globals
def get_data(n, T):
  return data_iidG_ER(n, T, p, q, r, s2, plt_ev = None,
                      ret_data = True)
def main(n):
  print 'n = %d' % n
  T = c*(n**2 - n) / 2
  pe = 0.0
  lmbda_star = 0.0
  rel_err_star = 0.0
  for k in range(K):
    D = get_data(n, T)['D']
    A_hat_k, lmbda_star_k,\
      rel_err_star_k = estimate_gcg(D, model, p, T, delta,
                                    ret_cv_result = True,
                                    numThreads = spams_threads)

    pe_k = edge_density(A_hat_k)
    pe += pe_k
    lmbda_star += lmbda_star_k
    rel_err_star += rel_err_star_k

  pe /= K
  lmbda_star /= K
  rel_err_star /= K
  return (pe, lmbda_star, rel_err_star, n)

if __name__ == '__main__':
  limit_memory_as(int(7000e6))
  np.random.seed(1)

  pool = Pool(num_threads)
  all_n = range(n_min, n_max + 1)
#  result = map(main, all_n)
  result = pool.map(main, all_n)
  result = sorted(result, key = lambda x: x[3])
  Pe, lmbda_star, rel_err = ([r[i] for r in result] for i in range(3))
  
  plt.plot(all_n, Pe, linewidth = 2, label = '$P_e$')
  plt.xlabel('$n$')
  plt.ylabel('Edge Proportion')
  plt.title('Edge Proportion vs Node Count (c = %d)' % c)
  plt.hlines(q, n_min, n_max, label = '$q$', linestyle = '--', linewidth = 2)
  plt.legend()
  plt.show()

  plt.plot(all_n, lmbda_star, linewidth = 2, label = '$\lambda^*$')
  plt.xlabel('$n$')
  plt.ylabel('$\lambda^*$')
  plt.title('$\lambda^*$ vs Node Count (c = %d)' % c)
  plt.legend()
  plt.show()

  plt.plot(all_n, rel_err, linewidth = 2, label = 'rel_err')
  plt.xlabel('$n$')
  plt.ylabel('rel_err')
  plt.title('Relative Error vs Node Count (c = %d)' % c)
  plt.legend()
  plt.show()
