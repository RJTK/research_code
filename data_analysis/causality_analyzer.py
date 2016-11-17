import sys

import pandas as pd
import numpy as np
import spams as spm

from scipy.sparse import bsr_matrix
from random import shuffle
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib import patches
from matplotlib.widgets import Slider
from progressbar import Bar, Percentage, ETA, ProgressBar, SimpleProgress

#My own stuff
from resource_limiter import limit_memory_as
from data_manipulation import *
from cross_validation import *
from ts_models import *

#This fits together various routines used for analyzing GC graphs.
#The data is expected to be already centered and stationary
#This is easy to garauntee since I'm using synthetic data...

DATA_DIR = '/home/ryan/Documents/academics/research/' \
           'granger_causality/software/datasets/synthetic/'

F_TRAIN = 0.7 #% for training
F_TEST = 0.9 #F_TEST - F_TRAIN = % for testing
F_VERIF = 1.0 #This should always be 1

#-------PLOTTING UTILITIES------------
def plot_matrix_ev(M, ax, mrkr = 'rx'):
  EVs = np.linalg.eigvals(M)
  ax.plot(EVs.real, EVs.imag, mrkr)
  return

#-------PACKAGED FITTING METHODS------------
def fit_var(A, OLS = True, OLST = True, LASSO = True, TRACE = True,
            DWGLASSO = True):
  G, D, p, n, T = A['G'], A['D'], A['p'], A['n'], A['T']

  #Baseline error measures
  def true_mean_err(Y):
    #Since the true mean is zero
    return np.linalg.norm(Y, 'fro')**2

  def training_mean_err(Y_train, Y_test):
    Y_mean = Y_train.mean(axis = 1)
    Y_mean = np.repeat(Y_mean, Y_test.shape[1]).reshape(Y_mean.shape[0],
                                                        Y_test.shape[1])
    err_mean_test = np.linalg.norm(Y_test - Y_mean, 'fro')**2
    return err_mean_test

  def prev_point_err(D, I):
    Y_prev, Y_hat_prev = build_YZ(D, I, 1)
    err_prev = np.linalg.norm(Y_prev - Y_hat_prev, 'fro')**2
    return err_prev

  def model_err(B, Y, Z):
    Y_hat = np.dot(B, Z)
    err = np.linalg.norm(Y_hat - Y, 'fro')**2
    return err

  #-----------Split up the data-------------
  D_train, I_train, D_test, I_test, D_verif, I_verif = split_data(
  D, T, F_TRAIN, F_TEST, F_VERIF)

  Y_all, Z_all = build_YZ(D, D.index, p)
  Y_train, Z_train = build_YZ(D_train, I_train, p)
  Y_test, Z_test = build_YZ(D_test, I_test, p)
  Y_verif, Z_verif = build_YZ(D_verif, I_verif, p)

  N_test = Y_test.size
  N_verif = Y_verif.size

  #Base line tests
  err_0_test = true_mean_err(Y_test) / N_test
  err_0_verif = true_mean_err(Y_verif) / N_verif
  err_mean_test = training_mean_err(Y_train, Y_test) / N_test
  err_mean_verif = training_mean_err(Y_train, Y_verif) / N_verif
  err_prev_test = prev_point_err(D_test, I_test) / N_test
  err_prev_verif = prev_point_err(D_verif, I_verif) / N_verif

  #Real methods
  if OLS:
    B_OLS = fit_ols(Y_train, Z_train)
    err_OLS_test = model_err(B_OLS, Y_test, Z_test) / N_test
    err_OLS_verif = model_err(B_OLS, Y_verif, Z_verif) / N_verif

  if OLST:
    B_OLST, lmbda_OLST_star, _ = cx_validate_opt(Y_train, Z_train,
                                                 Y_test, Z_test,
                                                 fit_olst,
                                                 lmbda_min = 0.0001,
                                                 lmbda_max = 5000)
    err_OLST_test = model_err(B_OLST, Y_test, Z_test) / N_test
    err_OLST_verif = model_err(B_OLST, Y_verif, Z_verif) / N_verif
    
  if LASSO:
    B_LASSO, lmbda_LASSO_star, _ = cx_validate_opt(Y_train, Z_train,
                                                   Y_test, Z_test,
                                                   spams_lasso,
                                                   lmbda_min = 0.0001,
                                                   lmbda_max = 5000)
    err_LASSO_test = model_err(B_LASSO, Y_test, Z_test) / N_test
    err_LASSO_verif = model_err(B_LASSO, Y_verif, Z_verif) / N_verif

  if TRACE:
    spams_trace = spams_trace_setup(Y_train, Z_train)
    B_TRACE, lmbda_TRACE_star, _ = cx_validate_opt(Y_train, Z_train,
                                                   Y_test, Z_test,
                                                   spams_trace,
                                                   lmbda_min = 0.0001,
                                                   lmbda_max = 5000)
    err_TRACE_test = model_err(B_TRACE, Y_test, Z_test) / N_test
    err_TRACE_verif = model_err(B_TRACE, Y_verif, Z_verif) / N_verif

  if DWGLASSO:
#    g = np.array([1] + [0]*(n**2 - 1))
    
#    DepthWiseGroups =  #DO THIS...
    spams_dwglasso = spams_glasso_setup(Y_train, Z_train)
    B_DWGLASSO, lmbda_DWGLASSO_star, _ = cx_validate_opt(Y_train, Z_train,
                                                         Y_test, Z_test,
                                                         spams_dwglasso,
                                                         lmbda_min = 0.0001,
                                                         lmbda_max = 5000)
    err_DWGLASSO_test = model_err(B_DWGLASSO, Y_test, Z_test) / N_test
    err_DWGLASSO_verif = model_err(B_DWGLASSO, Y_verif, Z_verif) / N_verif

  print 'err_0_test: %f' % err_0_test
  print 'err_mean_test: %f' % err_mean_test
  print 'err_prev_test: %f' % err_prev_test
  print 'err_OLS_test: %f' % err_OLS_test
  if OLST: print 'err_OLST_test: %f' % err_OLST_test
  if LASSO: print 'err_LASSO_test: %f' % err_LASSO_test
  if TRACE: print 'err_TRACE_test: %f' % err_TRACE_test
  if DWGLASSO: print 'err_DWGLASSO_test: %f' % err_DWGLASSO_test

  print '\n',

  print 'err_0_verif: %f' % err_0_verif
  print 'err_mean_verif: %f' % err_mean_verif
  print 'err_prev_verif: %f' % err_prev_verif
  print 'err_OLS_verif: %f' % err_OLS_verif
  if OLST: print 'err_OLST_verif: %f' % err_OLST_verif
  if LASSO: print 'err_LASSO_verif: %f' % err_LASSO_verif
  if TRACE: print 'err_TRACE_verif: %f' % err_TRACE_verif
  if DWGLASSO: print 'err_DWGLASSO_verif: %f' % err_DWGLASSO_verif

  results = {}
  if OLS: results['OLS'] = B_OLS
  if OLST: results['OLST'] = B_OLST
  if LASSO: results['LASSO'] = B_LASSO
  if TRACE: results['TRACE'] = B_TRACE
  if DWGLASSO: results['DWGLASSO'] = B_DWGLASSO

  return results

#------RESULTS---------
def ex1():
  np.random.seed(1)
  for pi in range(1, 4):
    for Ti in [200, 500, 1000]:
      A = load_data(DATA_DIR + 'iidG_ER_p%d_T%d.pkl' % (pi, Ti))
      print '----------DATA SET (p = %d, T = %d)----------' % (pi, Ti)
      result = fit_var(A)
      print '\n',
  return

def causality_graph_comparison(A):
  '''A comparison between LASS and OLST'''
  np.random.seed(1)
  G, D, p, n, T = A['G'], A['D'], A['p'], A['n'], A['T']

  D_train, I_train, D_test, I_test, D_verif, I_verif = split_data(
  D, T, F_TRAIN, F_TEST, F_VERIF)

  Y_all, Z_all = build_YZ(D, D.index, p)
  Y_train, Z_train = build_YZ(D_train, I_train, p)
  Y_test, Z_test = build_YZ(D_test, I_test, p)
  Y_verif, Z_verif = build_YZ(D_verif, I_verif, p)

  #--------FIT MODELS-------------
  B_OLST, lmbda_OLST_star, err_OLST_star = cx_validate_opt(Y_train, Z_train,
                                                           Y_test, Z_test,
                                                           fit_olst,
                                                           lmbda_min = 0.0001,
                                                           lmbda_max = 5000)
  B_LASSO, lmbda_LASSO_star, err_LASSO_star = cx_validate_opt(Y_train, Z_train,
                                                              Y_test, Z_test,
                                                              spams_lasso,
                                                              lmbda_min = 0.0001,
                                                              lmbda_max = 5000)

  #---------DISPLAY GRAPHS------------
  num_nodes = B_OLST.shape[0]
  A_OLST = sum(np.abs(B_OLST[0 : num_nodes,
                             k*num_nodes : (k + 1)*num_nodes])
               for k in range(0, p))
  A_OLST = (np.abs(A_OLST) > 0)

  #LASSO adj matrix
  A_LASSO = sum(np.abs(B_LASSO[0 : num_nodes,
                               k*num_nodes : (k + 1)*num_nodes])
                for k in range(0, p))
  A_LASSO = (np.abs(A_LASSO) > 0)

  fig, ax = plt.subplots(1, 3)
  ax[0].set_title('OLST, $\epsilon$ threshold')
  ax[1].set_title('LASSO')
  ax[2].set_title('Diff (Black indicates no difference)')
  plt.subplots_adjust(bottom = 0.3)
  A_OLST_img = ax[0].imshow(A_OLST, cmap = 'Greys')
  A_LASSO_img = ax[1].imshow(A_LASSO, cmap = 'Greys')
  diff_img = ax[2].imshow((A_OLST) == A_LASSO, cmap = 'Greys')

  eps_axis = plt.axes([.1, .05, .85, .05])
  lmbda_t = plt.axes([.1, .15, .85, .05])
  lmbda_l = plt.axes([.1, .25, .85, .05])
  eps_slider = Slider(eps_axis, '$\epsilon$', 0, np.max(B_OLST), valinit = 0)
  lmbda_t_slider = Slider(lmbda_t, '$\lambda_{l2}$', 0, 50*lmbda_OLST_star,
                          valinit = lmbda_OLST_star)

  lmbda_l_slider = Slider(lmbda_l, '$\lambda_{l1}$', 0, 50*lmbda_LASSO_star,
                          valinit = lmbda_LASSO_star)

  def update(x):
    eps = eps_slider.val
    lmbda_t = lmbda_t_slider.val
    lmbda_l = lmbda_l_slider.val

    B_OLST = fit_olst(Y_train, Z_train, lmbda_t)
    A_OLST = sum(np.abs(B_OLST[0 : num_nodes,
                               k*num_nodes : (k + 1)*num_nodes])
                 for k in range(0, p))
    A_OLST = np.array(np.abs(A_OLST) > eps, dtype = np.int)
    A_OLST_img.set_data(A_OLST)

    B_LASSO = spams_lasso(Y_train, Z_train, lmbda_l)
    A_LASSO = sum(np.abs(B_LASSO[0 : num_nodes,
                                 k*num_nodes : (k + 1)*num_nodes])
                  for k in range(0, p))
    A_LASSO = np.array(np.abs(A_LASSO) > 0, dtype = np.int)
    A_LASSO_img.set_data(A_LASSO)

    diff = np.array(A_OLST == A_LASSO, dtype = np.int)

    diff_img.set_data(diff)

    A_OLST_img.set_cmap('Greys')
    A_LASSO_img.set_cmap('Greys')
    diff_img.set_cmap('Greys')
    A_OLST_img.autoscale()
    A_LASSO_img.autoscale()
    diff_img.autoscale()
    fig.canvas.draw()
    return

  eps_slider.on_changed(update)
  lmbda_t_slider.on_changed(update)
  lmbda_l_slider.on_changed(update)

  plt.show()
  return

def ROC_curves(A, model):
  '''
  Plots various receiver operating characteristics
  '''
  G, D, p, n, T = A['G'], A['D'], A['p'], A['n'], A['T']
  B = A['B']

  D_train, I_train, D_test, I_test, D_verif, I_verif = split_data(
  D, T, F_TRAIN, F_TEST, F_VERIF)

  Y_train, Z_train = build_YZ(D_train, I_train, p)
  Y_test, Z_test = build_YZ(D_test, I_test, p)

  Lambda = np.linspace(0, 15, 100)
  widgets = [Percentage(), ' ', Bar(), ' ', ETA()]
  pbar = ProgressBar(widgets = widgets, maxval = len(Lambda))
  pbar.start()
  FP, TP = np.array([]), np.array([]) #False / True positive
  FN, TN = np.array([]), np.array([]) #False / True negative
  DP, DN = np.array([]), np.array([]) #Detected positive / Negative
  CP, CN = np.sum(G == 1), np.sum(G == 0) #Condition positive / negative

  for i, lmbda in enumerate(Lambda):
    pbar.update(i)
    B = spams_lasso(Y_train, Z_train, lmbda)
    A = sum(np.abs(B[0 : n, k*n : (k + 1)*n])
            for k in range(0, p))
    A = (np.abs(A) > 0)
    FP = np.append(FP, np.sum(np.logical_and(A == 1, G == 0)))
    TP = np.append(TP, np.sum(np.logical_and(A == 1, G == 1)))
    FN = np.append(FN, np.sum(np.logical_and(A == 0, G == 1)))
    TN = np.append(TN, np.sum(np.logical_and(A == 0, G == 0)))
    DP = np.append(P, np.sum(A == 1))
    DN = np.append(N, np.sum(A == 0))

  FDR = FP / P #False discovery rate
  TDR = TP / P #True discovery rate
  FOR = FN / N #False omission rate
  TOR = TN / N #True omission rate
    
  plt.plot(Lambda, FDR, label = 'FDR', linewidth = 2)
  plt.plot(Lambda, TDR, label = 'TDR', linewidth = 2)
  plt.plot(Lambda, FOR, label = 'FOR', linewidth = 2)
  plt.plot(Lambda, TOR, label = 'TOR', linewidth = 2)
  plt.legend()
  plt.title('Errors')
  plt.xlabel('$\lambda$')
  plt.ylabel('$P$')
  plt.show()

  plt.plot(FP / N, TP / P, linewidth = 2)
  plt.title('ROC Curve')
  plt.xlabel('FPR')
  plt.ylabel('TPR')
  plt.show()
  return

def causality_graph(A, model, FDR_plot = False):
  '''
  A should be a dataset dictionary (see data_synthesis/data_synth.py)
  model is one of the possible time series models, fit_ols, fit_olst etc...
  '''
  np.random.seed(1)
  G, D, p, n, T = A['G'], A['D'], A['p'], A['n'], A['T']
  B = A['B']

  D_train, I_train, D_test, I_test, D_verif, I_verif = split_data(
  D, T, F_TRAIN, F_TEST, F_VERIF)

  Y_train, Z_train = build_YZ(D_train, I_train, p)
  Y_test, Z_test = build_YZ(D_test, I_test, p)


  #--------FIT MODEL-------------
  B, lmbda_star, err_star = cx_validate_opt(Y_train, Z_train,
                                            Y_test, Z_test,
                                            model,
                                            lmbda_min = 0.0001,
                                            lmbda_max = 5000)

#  Lmbda = np.linspace(0.0001, 9, 500)
#  B, lmbda_star, errs = cx_validate(Y_train, Z_train,
#                                    Y_test, Z_test,
#                                    Lmbda, model)
#  err_star = min(errs)
  
  #adj matrix comparison
  cm =  mpl_colors.ListedColormap(['blue', 'white', 'black', 'red'])
  bounds = [-1.1, 0, 1, 2, 2.1]
  norm = mpl_colors.BoundaryNorm(bounds, cm.N)
  A = adj_matrix(B, p)

  fig, ax = plt.subplots(1, 1)
  ax.set_title(model.__name__)
  plt.subplots_adjust(bottom = 0.2)
  A_img = ax.imshow(2*A - G, cmap = cm, norm = norm,
                          interpolation = 'nearest')
  fig.colorbar(A_img, cmap = cm, norm = norm, boundaries = bounds,
               ticks = [-1, 0, 1, 2])
  lmbda = plt.axes([.1, .05, .85, .05])
  lmbda_slider = Slider(lmbda, '$\lambda$', 0, 50*lmbda_star,
                        valinit = lmbda_star)
  def update(x):
    lmbda = lmbda_slider.val
    B = model(Y_train, Z_train, lmbda)
    A = adj_matrix(B, p)
    A_img.set_data(2*A - G)
    A_img.autoscale()
    fig.canvas.draw()
    return

  lmbda_slider.on_changed(update)
  plt.show()
  return

def laplace_spectrum(DB, model):
  G, D, p, n, T = DB['G'], DB['D'], DB['p'], DB['n'], DB['T']
  B = DB['B']

  D_train, I_train, D_test, I_test, D_verif, I_verif = split_data(
  D, T, F_TRAIN, F_TEST, F_VERIF)

  Y_train, Z_train = build_YZ(D_train, I_train, p)
  Y_test, Z_test = build_YZ(D_test, I_test, p)


  #--------FIT MODEL-------------
  B, lmbda_star, err_star = cx_validate_opt(Y_train, Z_train,
                                            Y_test, Z_test,
                                            model,
                                            lmbda_min = 0.0001,
                                            lmbda_max = 5000)
  A_B = adj_matrix(B, p)
  L_B = lap_matrix(A_B)
  A_G = adj_matrix(G, 1)
  L_G = lap_matrix(A_G)

  fig, ax = plt.subplots(1,1)
  plt.subplots_adjust(bottom = 0.2)
  uc = patches.Circle((0, 0), radius = 1, fill = False,
                      color = 'black', ls = 'dashed', linewidth = 3)
  ax.add_patch(uc)
  ax.set_aspect('equal')
  ax.set_xlabel('Real')
  ax.set_ylabel('Imaginary')
  ax.set_title('Eigenvalues')

  EVs = np.linalg.eigvals(L_B)
  EV_plot = ax.scatter(EVs.real, EVs.imag)

#  plot_matrix_ev(L_B, ax, 'g+')
#  plot_matrix_ev(L_G, ax, 'rx')

  lmbda = plt.axes([.1, .05, .85, .05])
  lmbda_slider = Slider(lmbda, '$\lambda$', 0, 50*lmbda_star,
                        valinit = lmbda_star)
  def update(x):
    lmbda = lmbda_slider.val
    B = model(Y_train, Z_train, lmbda)
    A = adj_matrix(B, p)
    L = lap_matrix(A)
    EVs = np.linalg.eigvals(L)
    ax.scatter(EVs.real, EVs.imag)
    plt.show()
#    EV_plot.set_array(EVs.real, EVs.imag)
#    EV_plot.autoscale()
#    fig.canvas.draw()
    return

  lmbda_slider.on_changed(update)
  plt.show()

  return

if __name__ == '__main__':
  limit_memory_as(int(7000e6))
  np.random.seed(1)
  DB = load_data(DATA_DIR + 'iidG_ER_p1_T200.pkl')
#  DB = load_data(DATA_DIR + 'iidG_ER_p2_T500.pkl')
#  DB = load_data(DATA_DIR + 'iidG_ER_p2_T500_n5.pkl')
#  DB = load_data(DATA_DIR + 'iidG_ER_p2_T20000_n100.pkl')
#  causality_graph_comparison(DB)
#  causality_graph_LASSO(DB)
  spams_trace = spams_trace_setup()
  spams_glasso = spams_glasso_setup()

  fit_var_args = {'OLST' : True,
                  'LASSO' : True,
                  'TRACE' : True,
                  'DWGLASSO' : False}

  causality_graph(DB, fit_olst)
  fit_var(DB, **fit_var_args)
  causality_graph(DB, spams_lasso)
  causality_graph(DB, spams_trace)
  ROC_curves(DB, spams_lasso)
#  causality_graph_comparison(DB)
  laplace_spectrum(DB, spams_lasso)
