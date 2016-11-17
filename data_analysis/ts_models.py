import numpy as np
import spams as spm

from spams import lasso, fistaFlat
from scipy.optimize import minimize

from data_manipulation import *

SPAMS_TRACE_TOL = 5e-4

#-------TIME SERIES MODELS Y_t^ = sum_i B_i * Y_(t - i)----------
#Possible Additions:
#--Dantzig Selector (for n >> T)
#--Depth-Wise GLASSO
#--Time Varying LASSO
#--SLOPE (Candes)
def fit_ols(Y, Z):
  '''
  B = argmin_B ||Y - BZ||_F^2
  '''
  ZZT = np.dot(Z, Z.T)
  ZZT_inv = np.linalg.inv(ZZT)
  YZT = np.dot(Y, Z.T)
  B = np.dot(YZT, ZZT_inv)
  return B

def fit_olst(Y, Z, lmbda, ZZT = None):
  '''
  B = argmin_B ||Y - BZ||_F^2 + lmbda||B||_F^2
  '''
  if ZZT is None:
    ZZT = np.dot(Z, Z.T)
  tmp = lmbda*np.eye(ZZT.shape[0]) + ZZT
  tmp = np.linalg.inv(tmp)
  YZT = np.dot(Y, Z.T)
  B = np.dot(YZT, tmp)
  return B

def spams_trace_setup(Y = None, Z = None):
  '''
  Function closure to keep the previous B0 as the starting point for
  cross validation
  '''
  def spams_trace(Y, Z, lmbda):
    '''
    B = argmin_B ||Y - BZ||_F^2 + lmbda||B||_*
    '''
    if spams_trace.B0 is not None:
      B0 = spams_trace.B0
    else:
      B0 = fit_olst(Y, Z, lmbda)
      B0 = np.asfortranarray(B0.T)
      spams_trace.B0 = B0

    Y_spm = np.asfortranarray(Y.T)
    Z_spm = np.asfortranarray(Z.T)
    B, results = fistaFlat(Y_spm, Z_spm, B0, True, loss = 'square',
                           regul = 'trace-norm', lambda1 = lmbda,
                           verbose = False, tol = SPAMS_TRACE_TOL)
    spams_trace.B0 = B
    return B.T
  if Y is not None and Z is not None:
    B0 = fit_olst(Y, Z, 1)
    B0 = np.asfortranarray(B0.T)
    spams_trace.B0 = B0
  else:
    spams_trace.B0 = None
  return spams_trace

def spams_glasso_setup(Y = None, Z = None):
  '''
  Function closure to keep the previous B0 as the starting point for
  cross validation
  '''
  def spams_glasso(Y, Z, lmbda):
    '''
    b = argmin_b ||Y - [Z^T (x) I]b||_2^2 + lmbda sum_g\in G ||b_g||_2
    (Z^T (x) I) = vec(BZ) (kron product) this is needed to specify groups.
    '''
    if spams_glasso.b0 is not None:
      b0 = spams_glasso.b0
    else:
      B0 = fit_olst(Y, Z, lmbda)
      b0 = vec(B0)
      b0 = np.asfortranarray(b0.reshape((b0.shape[0], 1)))
      spams_glasso.b0 = b0

    n = Y.shape[0]
    y = vec(Y)
    Z = bsr_matrix(np.kron(np.eye(n), Z.T)) #This will be huge...

    y_spm = np.asfortranarray(y.reshape((y.shape[0], 1)))
    Z_spm = np.asfortranarray(Z)
    
    b, results = fistaFlat(y_spm, Z_spm, b0, True, loss = 'square',
                           regul = 'l1', lambda1 = lmbda,
                           verbose = False)
    spams_glasso.b0 = b
    B = np.array(b.reshape((n, n)))
    return B

  if Y is not None and Z is not None:
    B0 = fit_olst(Y, Z, 1)
    b0 = vec(B0)
    b0 = np.asfortranarray(b0.reshape((b0.shape[0], 1)))
    spams_glasso.b0 = b0
  else:
    spams_glasso.B0 = None
  return spams_glasso

def spams_lasso(Y, Z, lmbda):
  '''
  B = argmin_B ||Y - BZ||_F^2 + lmbda||B||_1
  '''
  Y_spm = np.asfortranarray(Y.T)
  Z_spm = np.asfortranarray(Z.T)
  B = lasso(Y_spm, Z_spm, lambda1 = lmbda, lambda2 = 0,
            mode = spm.PENALTY)
  B = B.toarray() #Convert from sparse array
  return B.T

def dwglasso_nm(Y, Z, lmbda):
  '''
  Implementation of the DWGLASSO method using simply the Nelder-Mead heuristic.
  This is unlikely to get particularly close to the minimum.
  '''
  
  return B