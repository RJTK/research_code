import pandas as pd
import numpy as np
import spams as spm

from random import shuffle
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from progressbar import Bar, Percentage, ETA, ProgressBar

#MY STUFF
from granger_models.ts_models import *
from granger_models.cross_validation import *
from granger_models.data_manipulation import *
from granger_models.resource_limiter import limit_memory_as
from stock_watson_data import load_data

SW_DIR = '/home/ryan/Documents/academics/research/'\
         'granger_causality/software/datasets/stock_watson/data/'
SW_MONTHLY_STATIONARY = 'stock_watson_monthly_stationary.csv'
SW_QUARTERLY_STATIONARY = 'stock_watson_quarterly_stationary.csv'
TEST_DATA = 'test_data.csv'

iso8601_date_parser = lambda d : pd.datetime.strptime(d, '%Y-%m-%d')

csv_params = {'header' : 0,
              'index_col' : 0,
              'parse_dates' : True,
              'date_parser' : iso8601_date_parser
}

#I could make some kind of wrapped sliding window
#In order to do better cross validation

F_TRAIN = 0.7 #% for training
F_TEST = 0.9 #F_TEST - F_TRAIN = % for testing
F_VERIF = 1.0 #This should always be 1

def model_err(B, Y, Z):
  Y_hat = np.dot(B, Z)
  err = np.linalg.norm(Y_hat - Y, 'fro')**2
  return err

def main():
  limit_memory_as(int(7000e6))
  np.random.seed(1)
  D = load_data(SW_DIR + SW_MONTHLY_STATIONARY, csv_params) #Data
  cols = list(D.columns)
  shuffle(cols)
  D = D[cols]
  
  T = len(D.index)

  #Split up the indices
  I_train = D.index[0:int(T*F_TRAIN)]
  I_test = D.index[int(T*F_TRAIN):int(T*F_TEST)]
  I_verif = D.index[int(T*F_TEST):int(T*F_VERIF)]

  T_train = len(I_train)
  T_test = len(I_test)
  T_verif = len(I_verif)

  #Split up the data
  D_train = D.ix[I_train].copy()
  D_test = D.ix[I_test].copy()
  D_verif = D.ix[I_verif].copy()

  #I should do some kind of reasonable lag order selection
  p = 5 #Lag order

  Y_all, Z_all = build_YZ(D, D.index, p)
  Y_train, Z_train = build_YZ(D_train, I_train, p)
  Y_test, Z_test = build_YZ(D_test, I_test, p)
  Y_verif, Z_verif = build_YZ(D_verif, I_verif, p)

  N_test = Y_test.size
  N_verif = Y_verif.size

  #-------TRUE MEAN--------
  #Since the mean is 0
  err_0_test = np.linalg.norm(Y_test, 'fro')**2 / N_test 
  err_0_verif = np.linalg.norm(Y_verif, 'fro')**2 / N_verif
  #-----------------------
  
  #-------TRAINING MEAN------
  Y_mean = Y_train.mean(axis = 1)
  Y_mean_test_hat = np.repeat(Y_mean, Y_test.shape[1]).reshape(Y_mean.shape[0],
                                                               Y_test.shape[1])
  Y_mean_verif_hat = np.repeat(Y_mean, Y_verif.shape[1]).reshape(Y_mean.shape[0],
                                                                 Y_verif.shape[1])
  err_mean_test = np.linalg.norm(Y_test - Y_mean_test_hat, 'fro')**2 / N_test
  err_mean_verif = np.linalg.norm(Y_verif - Y_mean_verif_hat, 'fro')**2 / N_verif
  #--------------------------

  #------PREVIOUS DATA POINT---------
  Y_prev_test, Y_prev_test_hat = build_YZ(D_test, I_test, 1)
  Y_verif_test, Y_verif_test_hat = build_YZ(D_verif, I_verif, 1)
  err_prev_test = np.linalg.norm(Y_prev_test - Y_prev_test_hat, 'fro')**2 / N_test
  err_prev_verif = np.linalg.norm(Y_verif_test - 
                                  Y_verif_test_hat, 'fro')**2 / N_verif
  #----------------------------------

  #----------OLS----------
  B_OLS = fit_ols(Y_train, Z_train)
  err_OLS_test = model_err(B_OLS, Y_test, Z_test) / N_test
  err_OLS_verif = model_err(B_OLS, Y_verif, Z_verif) / N_verif
  Y_OLS_test_hat = np.dot(B_OLS, Z_test)
  #-----------------------

  #-------OLST------------
  B_OLST, lmbda_OLST_star, errs_OLST = cx_validate_opt(Y_train, Z_train,
                                                       Y_test, Z_test,
                                                       fit_olst,
                                                       lmbda_min = 0.0001,
                                                       lmbda_max = 5000)
  err_OLST_test = model_err(B_OLST, Y_test, Z_test) / N_test
  err_OLST_verif = model_err(B_OLST, Y_verif, Z_verif) / N_verif
  Y_OLST_test_hat = np.dot(B_OLST, Z_test)
  #-----------------------

  #-------LASSO-----------
  B_LASSO, lmbda_LASSO_star, _ = cx_validate_opt(Y_train, Z_train,
                                                 Y_test, Z_test,
                                                 spams_lasso,
                                                 lmbda_min = 0.0001,
                                                 lmbda_max = 5000)
  err_LASSO_test = model_err(B_LASSO, Y_test, Z_test) / N_test
  err_LASSO_verif = model_err(B_LASSO, Y_verif, Z_verif) / N_verif
  Y_LASSO_test_hat = np.dot(B_LASSO, Z_test)
  #----------------------

  #-------DWGLASSO-------
  B_DWGLASSO, lmbda_DWGLASSO_star, _ = cx_validate_opt(Y_train, Z_train,
                                                       Y_test, Z_test,
                                                       dwglasso,
                                                       lmbda_min = 0.0001,
                                                       lmbda_max = 5000)
  err_DWGLASSO_test = model_err(B_DWGLASSO, Y_test, Z_test) / N_test
  err_DWGLASSO_verif = model_err(B_DWGLASSO, Y_verif, Z_verif) / N_verif
  Y_DWGLASSO_test_hat = np.dot(B_DWGLASSO, Z_test)
  #----------------------

  print 'err_0_test: %f' % err_0_test
  print 'err_mean_test: %f' % err_mean_test
  print 'err_prev_test: %f' % err_prev_test
  print 'err_OLS_test: %f' % err_OLS_test
  print 'err_OLST_test: %f' % err_OLST_test
  print 'err_LASSO_test: %f' % err_LASSO_test
  print 'err_DWGLASSO_test: %f' % err_DWGLASSO_test

  print '\n',

  print 'err_0_verif: %f' % err_0_verif
  print 'err_mean_verif: %f' % err_mean_verif
  print 'err_prev_verif: %f' % err_prev_verif
  print 'err_OLS_verif: %f' % err_OLS_verif
  print 'err_OLST_verif: %f' % err_OLST_verif
  print 'err_LASSO_verif: %f' % err_LASSO_verif
  print 'err_DWGLASSO_verif: %f' % err_DWGLASSO_verif

  #-----------PLOTS OF THE PREDICTIONS----------------
  plt.plot(Y_test[0, :], linewidth = 2, label = 'True')
  plt.plot(Y_OLST_test_hat[0, :], linewidth = 2, label = 'OLST')
  plt.plot(Y_LASSO_test_hat[0, :], linewidth = 2, label = 'LASSO')
  plt.plot(Y_DWGLASSO_test_hat[0, :], linewidth = 2, label = 'DWGLASSO')
  plt.legend()
  plt.ylabel('$X_t$')
  plt.xlabel('$t$')
  plt.title('Predictions')
  plt.show()
  #---------------------------------------------------

  #-----------------ADJACENCY MATRICES----------------
  #OLST adj matrix
  A_OLST = adj_matrix(B_OLST, p, delta = 0.1) #Totally arbitrary delta

  #LASSO adj matrix
  A_LASSO = adj_matrix(B_LASSO, p, delta = 0)

  #DWGLASSO adj matrix
  A_DWGLASSO = adj_matrix(B_DWGLASSO, p, delta = 1e-6)

  fig, ax = plt.subplots(1,3)
  ax[0].set_title('OLST, $\epsilon$ threshold')
  ax[1].set_title('LASSO')
  ax[2].set_title('Diff')
  plt.subplots_adjust(bottom = 0.3)
  A_OLST_img = ax[0].imshow(A_OLST, cmap = 'Greys')
  A_LASSO_img = ax[1].imshow(A_LASSO, cmap = 'Greys')
  diff_img = ax[2].imshow((A_OLST) == A_LASSO, cmap = 'Greys')

  eps_axis = plt.axes([.1, .05, .85, .05])
  lmbda_t = plt.axes([.1, .15, .85, .05])
  lmbda_l = plt.axes([.1, .25, .85, .05])
  eps_slider = Slider(eps_axis, '$\epsilon$', 0, np.max(B_OLST), valinit = 0)
  lmbda_t_slider = Slider(lmbda_t, '$\lambda_{l2}$', 0, 10*np.max(Lmbda),
                          valinit = lmbda_OLST_star)
  lmbda_l_slider = Slider(lmbda_l, '$\lambda_{l1}$', 0, np.max(Lmbda),
                          valinit = lmbda_LASSO_star)

  def update(x):
    eps = eps_slider.val
    lmbda_t = lmbda_t_slider.val
    lmbda_l = lmbda_l_slider.val

    B_OLST = OLS_tikhonov(Y_train, Z_train, lmbda_t)
    A_OLST = sum(np.abs(B_OLST[0 : num_nodes,
                               k*num_nodes : (k + 1)*num_nodes])
                 for k in range(0, p))
    A_OLST_img.set_data(np.abs(A_OLST) > eps)

    B_LASSO = spams_lasso(Y_train, Z_train, lmbda_l)
    A_LASSO = sum(np.abs(B_LASSO[0 : num_nodes,
                                 k*num_nodes : (k + 1)*num_nodes])
                  for k in range(0, p))
    A_LASSO = (np.abs(A_LASSO) > 0)
    A_LASSO_img.set_data(A_LASSO)

#    diff_img.set_data((A_OLST < eps) == A_LASSO)
    diff_img.set_data(np.logical_and(A_LASSO > 0, A_LASSO.T > 0))

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

  #--------------SYMMETRIC-----------

  A_LASSO = np.array((np.abs(A_LASSO) + np.abs(A_LASSO.T)) > 0, dtype = np.float64)
  D_LASSO = np.diag(np.sum(A_LASSO, axis = 1))
  L_LASSO = D_LASSO - A_LASSO + np.diag(np.diag(A_LASSO))
  L_LASSO = np.dot(np.dot(1./np.sqrt(D_LASSO), L_LASSO), 1./np.sqrt(D_LASSO))

  #Black: edge, White: no edge
  plt.imshow(np.abs(A_LASSO) == 0, cmap = 'Greys_r')
  plt.show()

  plt.imshow(D_LASSO)
  plt.colorbar()
  plt.show()

  plt.imshow(L_LASSO)
  plt.colorbar()
  plt.show()

  ev = np.linalg.eigvals(L_LASSO)
  plt.hist(ev)
  plt.show()

  '''
  Y_hat_all_OLST = np.dot(B_OLST, Z_all)
  Y_hat_all_LASSO = np.dot(B_LASSO, Z_all)
  for i in range(18):
    plt.plot(Y_all[:, 6*i], color = 'r', linewidth = 2, label = 'Y')
    plt.plot(Y_hat_all_OLST[:, 2*i], color = 'm', linewidth = 2, label = 'Y_hat_OLST')
    plt.plot(Y_hat_all_LASSO[:, 2*i], color = 'g', linewidth = 2, label = 'Y_hat_LASSO')
    plt.title(D.columns[i])
    plt.xlabel('$t$')
    plt.ylabel('$X_t$')
    plt.legend()
    plt.show()
  '''
  return

if __name__ == '__main__':
  main()
