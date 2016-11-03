import pandas as pd
import numpy as np
import spams as spm

from random import shuffle
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from spams import lasso
from progressbar import Bar, Percentage, ETA, ProgressBar

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

def build_YZ(D, I, p):
  T = len(I)
  if T == 0:
    return np.array([]), np.array([])
  Y = np.array(D[p:]).T

  Z = np.array(D.ix[I[p - 1: : -1]]).flatten()
  for k in range(1, T - p):
    Zk = np.array(D.ix[I[k + p - 1: k - 1: -1]]).flatten()
    Z = np.vstack((Z, Zk))
  Z = Z.T
  return Y, Z

#This is the solution for the formulation
#||Y - BZ||_F^2 Where B is the variable
def OLS(Y, Z):
  ZZT = np.dot(Z, Z.T)
  ZZT_inv = np.linalg.inv(ZZT)
  YZT = np.dot(Y, Z.T)
  B = np.dot(YZT, ZZT_inv)
  return B

def OLS_tikhonov(Y, Z, lmbda, ZZT = None):
  if ZZT is None:
    ZZT = np.dot(Z, Z.T)
  tmp = lmbda*np.eye(ZZT.shape[0]) + ZZT
  tmp = np.linalg.inv(tmp)
  YZT = np.dot(Y, Z.T)
  B = np.dot(YZT, tmp)
  return B

def spams_lasso(Y, Z, lmbda):
  Y_spm = np.asfortranarray(Y.T)
  Z_spm = np.asfortranarray(Z.T)
  B = lasso(Y_spm, Z_spm, lambda1 = lmbda, lambda2 = 0,
            mode = spm.PENALTY)
  B = B.toarray()
  return B.T

def cx_validate(Y_train, Z_train, Y_test, Z_test, Lmbda, f, **kwargs):
  errs = []
  widgets = [Percentage(), ' ', Bar(), ' ', ETA()]
  pbar = ProgressBar(widgets = widgets, maxval = len(Lmbda))
  pbar.start()
  for lmbda_i, lmbda in enumerate(Lmbda):
    pbar.update(lmbda_i)
    B = f(Y_train, Z_train, lmbda, **kwargs)
    Y_hat_test = np.dot(B, Z_test)
    err_test = np.linalg.norm(Y_test - Y_hat_test, 'fro')**2
    errs.append(err_test)
    try:
      if(err_test < min_err):
        B_star = B
        min_err = err_test
        lmbda_star = lmbda
    except NameError:
      B_star = B
      min_err = err_test
      lmbda_star = lmbda
  pbar.finish()
  return B_star, lmbda_star, errs

def main():
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
  p = 2 #Lag order

  Y_all, Z_all = build_YZ(D, D.index, p)
  Y_train, Z_train = build_YZ(D_train, I_train, p)
  Y_test, Z_test = build_YZ(D_test, I_test, p)
  Y_verif, Z_verif = build_YZ(D_verif, I_verif, p)

  #-------TRUE MEAN--------
  err_0_test = np.linalg.norm(Y_test, 'fro')**2 #Since the mean is 0
  #-----------------------
  
  #-------TRAINING MEAN------
  Y_mean = Y_train.mean(axis = 1)
  Y_mean = np.repeat(Y_mean, Y_test.shape[1]).reshape(Y_mean.shape[0],
                                                      Y_test.shape[1])
  err_mean_test = np.linalg.norm(Y_test - Y_mean, 'fro')**2
  #--------------------------

  #------PREVIOUS DATA POINT---------
  Y_prev_test, Y_hat_prev_test = build_YZ(D_test, I_test, 1)
  err_prev_test = np.linalg.norm(Y_prev_test - Y_hat_prev_test, 'fro')**2
  #----------------------------------

  #------RANDOM WALK--------
  #DO THIS
  #-------------------------

  #----------OLS----------
  B_OLS = OLS(Y_train, Z_train)
  Y_hat_OLS_test = np.dot(B_OLS, Z_test)
  err_OLS_test = np.linalg.norm(Y_test - Y_hat_OLS_test, 'fro')**2
  #-----------------------

  #NOTE: This is the maximum penalty for LASSO, it can only really be
  #Heuristic for OLST
  lmbda_max = np.max(np.dot(Z_train, Y_train.T))
  print 'lmbda_max: %f' % lmbda_max
  Lmbda = np.linspace(0, lmbda_max, 30)

  #-------OLST------------
  ZZT = np.dot(Z_train, Z_train.T)
  B_OLST, lmbda_OLST_star, errs_OLST = cx_validate(Y_train, Z_train, Y_test,
                                                   Z_test, Lmbda,
                                                   OLS_tikhonov, ZZT = ZZT)
  Y_hat_OLST_test = np.dot(B_OLST, Z_test)
  err_OLST_test = np.linalg.norm(Y_hat_OLST_test - Y_test, 'fro')**2
  #-----------------------

  #-------LASSO-----------
  B_LASSO, lmbda_LASSO_star, errs_LASSO = cx_validate(Y_train, Z_train, Y_test,
                                                      Z_test, Lmbda, spams_lasso)
  Y_hat_LASSO_test = np.dot(B_LASSO, Z_test)
  err_LASSO_test = np.linalg.norm(Y_hat_LASSO_test - Y_test, 'fro')**2
  #----------------------

  #-----------PLOTS OF THE PREDICTIONS----------------
  plt.plot(Y_test[0, :], linewidth = 2, label = 'True')
  plt.plot(Y_hat_prev_test[0, :], linewidth = 2, label = 'Prev')
  plt.plot(Y_hat_OLST_test[0, :], linewidth = 2, label = 'OLST')
  plt.plot(Y_hat_LASSO_test[0, :], linewidth = 2, label = 'LASSO')
  plt.legend()
  plt.ylabel('$X_t$')
  plt.xlabel('$t$')
  plt.title('Predictions')
  plt.show()
  #---------------------------------------------------

#  plt.hlines(err_OLS_test, min(Lmbda), max(Lmbda), linewidth = 2,
#             label = 'OLS', color = 'g')
  plt.hlines(err_mean_test, min(Lmbda), max(Lmbda), linewidth = 2,
             label = 'mean', color = 'r')
  plt.hlines(err_0_test, min(Lmbda), max(Lmbda), linewidth = 2,
             label = '0', color = 'm')
  plt.plot(Lmbda, errs_OLST, linewidth = 2, label = 'Tikhonov')
  plt.plot(Lmbda, errs_LASSO, linewidth = 2, label = 'LASSO')
  plt.legend()
  plt.ylabel('MSE_train')
  plt.xlabel('$\lambda$')
  plt.show()

#  err_OLST_test = min(errs_OLST)
#  err_LASSO_test = min(errs_LASSO)
  N = Y_test.size
  print 'err_0_test: %f' % (err_0_test / N)
  print 'err_mean_test: %f' % (err_mean_test / N)
  print 'err_prev_test: %f' % (err_prev_test / N)
  print 'err_OLS_test: %f' % (err_OLS_test / N)
  print 'err_OLST_test: %f' % (err_OLST_test / N)
  print 'err_LASSO_test: %f' % (err_LASSO_test / N)

  err_0_verif = np.linalg.norm(Y_verif, 'fro')**2
  
  Y_mean = Y_train.mean(axis = 1)
  Y_mean = np.repeat(Y_mean, Y_verif.shape[1]).reshape(Y_mean.shape[0],
                                                      Y_verif.shape[1])
  err_mean_verif = np.linalg.norm(Y_verif - Y_mean, 'fro')**2

  Y_mean_true = Y_verif.mean(axis = 1)
  Y_mean = np.repeat(Y_mean_true, Y_verif.shape[1]).reshape(Y_mean.shape[0],
                                                      Y_verif.shape[1])
  err_mean_true_verif = np.linalg.norm(Y_verif - Y_mean, 'fro')**2

  Y_hat_OLS_verif = np.dot(B_OLS, Z_verif)
  err_OLS_verif = np.linalg.norm(Y_verif - Y_hat_OLS_verif, 'fro')**2

  Y_hat_OLST_verif = np.dot(B_OLST, Z_verif)
  err_OLST_verif = np.linalg.norm(Y_verif - Y_hat_OLST_verif, 'fro')**2

  Y_hat_LASSO_verif = np.dot(B_LASSO, Z_verif)
  err_LASSO_verif = np.linalg.norm(Y_verif - Y_hat_LASSO_verif, 'fro')**2

  N = Y_verif.size
  print 'err_0_verif: %f' % (err_0_verif / N)
  print 'err_mean_verif: %f' % (err_mean_verif / N)
  print 'err_mean_true_verif: %f' % (err_mean_true_verif / N)
  print 'err_OLS_verif: %f' % (err_OLS_verif / N)
  print 'err_OLST_verif: %f' % (err_OLST_verif / N)
  print 'err_LASSO_verif: %f' % (err_LASSO_verif / N)

  #-----------------ADJACENCY MATRICES----------------

  #OLST adj matrix
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
