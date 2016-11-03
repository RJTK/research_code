import numpy as np
import scipy as sp

from matplotlib import pyplot as plt
from matplotlib import patches

#---------MEMORY LIMIT----------
import os
import psutil

def memory_usage_MB():
  '''Returns in MB the current virtual memory size'''
  denom = 1e6
  proc = psutil.Process(os.getpid())
  mem = proc.memory_info().vms
  return mem / denom

def memory_usage_pct():
  '''Returns in percent the current data memory size'''
  proc = psutil.Process(os.getpid())
  mempct = proc.memory_percent(memtype = 'vms')
  return mempct

#Seems to be different units from what the mem usages return?
def limit_memory_as(soft_lim):
  '''Sets the (soft) limit to the process's virtual memory usage'''
  proc = psutil.Process(os.getpid())
  _, hard_lim = proc.rlimit(psutil.RLIMIT_AS)
  proc.rlimit(psutil.RLIMIT_AS, (soft_lim, hard_lim))
  return

#--------------------------------

def plot_matrix_ev(M, ax, mrkr = 'rx'):
  EVs = np.linalg.eigvals(M)
  for ev in EVs:
    ax.plot(ev.real, ev.imag, mrkr)
  return

def random_matrix(n, d):
  M = d(n)
  return M

def random_matrix_sum(n, d, k):
  M = d(n)
  for i in range(k - 1):
    M += d(n)
  return M

def block_companion(B):
  '''
  Produces a block companion from the matrices B[0], B[1], ... , B[p - 1]
  '''
  p = len(B)
  B = np.hstack((B[k] for k in range(p))) #The top row
  n = B.shape[0]

  I = np.eye(n*(p - 1))
  Z = np.zeros((n*(p - 1), n))
  R = np.hstack((I, Z))
  B = np.vstack((B, R))

  return B

if __name__ == '__main__':
  limit_memory_as(int(7000e6))

  mu = 0 #Mean seems to have no effect, even when P increases
  P = 1 #Delay length
  r = 1 #Expected gershgorin circle radius
  stdev = 1 #increasing this spreads out the evs
  n = 100 #Matrix size
  N = 50 #Num of Matrices

  p = .85
  dN = lambda n : np.random.normal(mu, stdev, size = (n, n))
#  dN = lambda n : np.random.exponential(scale = 5, size = (n, n))
  dB = lambda n : np.random.binomial(n = 1, p = p, size = (n, n))

#  rand_matrix = lambda : (random_matrix(n, dB) * random_matrix(n, dN)) \
#                / np.sqrt(n)
  rand_matrix = lambda : ((np.pi / 2)**(0.25)) * np.sqrt(float(r) / (n - 1)) * \
                (random_matrix(n, dN))

  M = [rand_matrix() for k in range(P)]
  B = block_companion(M)

  plt.imshow(B)
  plt.colorbar()
  plt.show()

  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  uc = patches.Circle((0, 0), radius = 1, fill = False,
                      color = 'black', ls = 'dashed', linewidth = 3)
  Pc = patches.Circle((0, 0), radius = np.sqrt(p), fill = False,
                      color = 'black', ls = 'solid', linewidth = 3)
  ax.add_patch(uc)
  ax.add_patch(Pc)
  ax.set_aspect('equal')
  ax.set_xlabel('Real')
  ax.set_ylabel('Imaginary')
  ax.set_title('Eigenvalues')
  ax.text(0.8, 1.35, '$M_{ij} \in N(%d, %d)$' % (mu, stdev), fontsize = 18)
  ax.text(1.0, 1.2, '$n = %d$' % n, fontsize = 18)
  ax.text(1.0, 1.05, '$P = %d$' % P, fontsize = 18)
#  ax.text(.5, .6, '$p = %f$' % p, fontsize = 18)

  ax.set_xlim([-1.5, 1.5])
  ax.set_ylim([-1.5, 1.5])

  for i in range(N):
    M = [rand_matrix() for k in range(P)]
    B = block_companion(M)
    #M = random_matrix_sum(n, dN, K) / (np.sqrt(K) * stdev * np.sqrt(n))
    #B = random_matrix(n, dB)
    #M = M*B
    plot_matrix_ev(B, ax, 'g+')

  plt.show()
