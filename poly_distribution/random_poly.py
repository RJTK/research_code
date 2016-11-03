import numpy as np
import scipy as sp

from matplotlib import pyplot as plt
from matplotlib import patches

def plot_poly_roots(p, ax, mrkr = 'rx'):
  '''Plots the roots of p on the axis ax'''
  roots = np.roots(p)
  for r in roots:
    ax.plot(r.real, r.imag, mrkr)
  return

def random_poly_byc(n, d):
  '''
  Creates a monic polynomial with coefficients randomly drawn
  via d(n).  That is, d should be a random sampling function
  '''
  c = d(n)
  p = np.poly1d(c / c[0])
  return p

if __name__ == '__main__':
  mu = 0
  stdev = 1
  n = 11 #Poly degree
  N = 1000 #Number of polynomials

  d = lambda n : np.random.normal(mu, stdev, n)
#  d = lambda n : np.random.uniform(low = -5, high = 5, size = n)
#  d = lambda n : np.random.exponential(scale = 3, size = n)
#  d = lambda n : np.random.standard_cauchy(n)

  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  uc = patches.Circle((0, 0), radius = 1, fill = False,
                      color = 'black', ls = 'dashed')
  ax.add_patch(uc)
  ax.set_aspect('equal')
  ax.set_xlabel('Real')
  ax.set_ylabel('Imaginary')
  ax.set_title('Poly Roots')
  ax.text(1.0, 1.7, '$P_i \in N(%d, %d)$' % (mu, stdev), fontsize = 18)
  ax.text(1.1, 1.5, '$d = %d$' % n, fontsize = 18)

  ax.set_xlim([-2, 2])
  ax.set_ylim([-2, 2])

  for i in range(N):
    p = random_poly_byc(n, d)
    plot_poly_roots(p, ax, 'g+')

  plt.show()
