'''DATA SOURCE: http://www.princeton.edu/~mwatson/publi.html'''
import math

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

import warnings
warnings.filterwarnings(action = 'ignore', category = ImportWarning)
#warnings.filterwarnings(action = 'error', category = 

#Stationarity transform functions
#the argument should be a 
tcode1 = lambda x : x #identity function
tcode2 = lambda x : pd.Series.diff(x) #single difference
tcode3 = lambda x : pd.Series.diff(pd.Series.diff(x)) #second difference
tcode4 = lambda x : np.log(x)
tcode5 = lambda x : pd.Series.diff(np.log(x))
tcode6 = lambda x : pd.Series.diff(pd.Series.diff(np.log(x)))

tcode1_inv = lambda x : x
tcode2_inv = lambda x : pd.Series.cumsum(x)
tcode3_inv = lambda x : pd.Series.cumsum(pd.Series.cumsum(x))
tcode4_inv = lambda x : np.exp(x)
tcode5_inv = lambda x : np.exp(pd.Series.cumsum(x))
tcode6_inv = lambda x : np.exp(pd.Series.cumsum(pd.Series.cumsum(x)))

xform_funcs = [tcode1, tcode2, tcode3, tcode4, tcode5, tcode6]
inv_xform_funcs = [tcode1_inv, tcode2_inv, tcode3_inv,
                   tcode4_inv, tcode5_inv, tcode6_inv]

#Stock and Watson data (143 macroeconomic quarterly series 1960-2008)
SW_DIR = '/home/ryan/Documents/academics/research/'\
         'granger_causality/software/datasets/stock_watson/data/'
SW_MONTHLY = 'es09_1_sh1.csv' #Monthly data
SW_QUARTERLY = 'es09_1_sh2.csv' #Quarterly data

#OUTPUT FILES
SW_MONTHLY_STATIONARY = 'stock_watson_monthly_stationary.csv'
SW_QUARTERLY_STATIONARY = 'stock_watson_quarterly_stationary.csv'

#If true, apply different Tcodes from stock and watson
MY_TCODES = False

#Date parser for stock and watson data
sw_date_parser = lambda d : pd.datetime.strptime(d, '%m\%d\%Y')

#Params for loading the data
monthly_csv_params = {'header' : 0,
                      'skiprows' : range(1,10),
                      'index_col' : 0,
                      'parse_dates' : True,
                      'date_parser' : sw_date_parser
}

quarterly_csv_params = {'header' : 0,
                        'skiprows' : range(1, 9),
                        'index_col' : 0,
                        'parse_dates' : True,
                        'date_parser' : sw_date_parser
}

#Params for output csv file
output_csv_params = {'header' : True,
                     'index' : True,
                     'encoding' : 'ascii',
                     'date_format' : '%Y-%m-%d'
}

#Params for transform functions
monthly_csv_extras_params = {'header' : 0,
                             'index_col' : 0,
                             'nrows' : 9
}

quarterly_csv_extras_params = {'header' : 0,
                              'index_col' : 0,
                              'nrows' : 8
}

#Params for pystatsmodels Augmented Dickey Fuller test
adfuller_params = {'regression' : 'nc',
                   'autolag' : 'AIC'
}

def load_data(file_name, csv_params):
  df = pd.read_csv(file_name, **csv_params)

  #Strip whitespace from labels
  df.rename(columns = {hdr : str.strip(hdr)
                       for hdr in df.columns},
            inplace = True)

  #There are columns that are for some reason being read as an object
  for s in df.select_dtypes(include = ('object',)):
    df.loc[:, s] = pd.to_numeric(df[s], errors = 'coerce')

  return df

def load_params(file_name, csv_params):
  params = pd.read_csv(file_name, **csv_params)
  #Strip whitespace
  params.rename(columns = {hdr : str.strip(hdr)
                           for hdr in params.columns},
                inplace = True)
  params = params.T
  return params

def data_xform(df, df_Tcodes, xform_funcs, xform_set = None):
  if xform_set is None:
    xform_set = df_Tcodes.index

  #Make a copy of the dataframe which will be returned.
  #This allows one to call for example s = data_xform(q)
  df_cpy = df.copy()

  for s in xform_set:
    t = df_Tcodes[s]
    if t >= 3:
      assert (df[s] > 0).all(), \
        'Series %s with Tcodes %d is not strictly positive!' % (s, t)
      #Index the appropriate function from the table
    df_cpy.loc[:, s] = xform_funcs[t - 1](df[s])

  df_cpy = df_cpy.dropna()
  return df_cpy

#These functions depend on the form of the other!
#The order of subtracting the mean and multiplying
#by the variance matters!
#--------------------------------------------
def normalize_and_center(df, df_params):
  df_params_cpy = df_params.copy()
  df_params_cpy.loc[:, 'mean'] = df.mean(0)
  df_params_cpy.loc[:, 'std'] = df.std(0, ddof = 0)
  df_cpy = df - df_params_cpy['mean']
  df_cpy = df_cpy / df_params_cpy['std']
  return df_cpy, df_params_cpy

def revert_normalize_and_center(df, df_params):
  df_cpy = df * df_params.ix['std']
  df_cpy = df + df_params.ix['mean']
  return df_cpy
#--------------------------------------------

def test_stationarity(df, adf_params, adf_pvalue = 0.05):
  nonstationary_series = []

  for s in df.columns:
    result = adfuller(df[s], **adf_params)
    pv = result[1] #Approximate p-value
    if pv >= adf_pvalue:
      nonstationary_series.append((s, pv))

  nonstationary_series = sorted(nonstationary_series,
                                reverse = True,
                                key = lambda tup : tup[1])
  return nonstationary_series

def update_Tcodes(nonstationary, df_Tcodes):
  #nonstationary is a list of tuples (name, pv)
  df_Tcodes_cpy = pd.DataFrame(df_Tcodes)
  for s in nonstationary:
    name = s[0]
    tcode = df_Tcodes[name]
    if tcode == 1:
      df_Tcodes_cpy.loc[name] = 2
    elif tcode == 2:
      df_Tcodes_cpy.loc[name] = 3
    elif tcode == 3:
      print 'WARN: series %s already 2nd differenced' % name
    elif tcode == 4:
      df_Tcodes_cpy.loc[name] = 5
    elif tcode == 5:
      df_Tcodes_cpy.loc[name] = 6
    elif tcode == 6:
      print 'WARN: series %s already 2nd differenced' % name
    else:
      raise AssertionError('Tcode invalid!')
  return df_Tcodes

def main():
  #Load data and Tcodes
  monthly_original = load_data(SW_DIR + SW_MONTHLY, monthly_csv_params)
  monthly_params_original = load_params(SW_DIR + SW_MONTHLY,
                                        monthly_csv_extras_params)
  quarterly_original = load_data(SW_DIR + SW_QUARTERLY, quarterly_csv_params)
  quarterly_params_original = load_params(SW_DIR + SW_QUARTERLY,
                                          quarterly_csv_extras_params)

  #Copy all the data
  monthly = pd.DataFrame(monthly_original)
  monthly_params = pd.DataFrame(monthly_params_original)
  quarterly = pd.DataFrame(quarterly_original)
  quarterly_params = pd.DataFrame(quarterly_params_original)

  #These cause some problems because SW suggests taking logs of them...
  #And their values go negative during the '08 crisis
  #They are institutional reserve quantities
#  monthly = monthly.drop('FMRNBA', 1)
#  monthly_params = monthly_params.drop('FMRNBA', 1)

  monthly_params.loc[:, 'Tcode'] = monthly_params['Tcode'].apply(int)
  quarterly_params.loc[:, 'Tcode'] = quarterly_params['Tcode'].apply(int)

  #See comment above
  monthly_params.loc['FMRNBA', 'Tcode'] = 1

  #Apply stationarity transforms
  monthly = monthly.dropna()
  quarterly = quarterly.dropna()
  monthly = data_xform(monthly, monthly_params['Tcode'], xform_funcs)
  quarterly = data_xform(quarterly, quarterly_params['Tcode'], xform_funcs)

  #Normalize and center the data
  monthly, monthly_params = normalize_and_center(monthly, monthly_params)
  quarterly, quarterly_params = normalize_and_center(quarterly, quarterly_params)

  #--------FIX MONTHLY STATIONARITY------------
  #Test stationarity
  pv = .05
  monthly_nonstationary = test_stationarity(monthly, adfuller_params, pv)
  print 'Number of nonstationary monthly series (SW Tcodes) at p ~= %f: %d'\
    % (pv, len(monthly_nonstationary))

  pv = .01
  monthly_nonstationary = test_stationarity(monthly, adfuller_params, pv)
  print 'Number of nonstationary monthly series (SW Tcodes) at p ~= %f: %d'\
    % (pv, len(monthly_nonstationary))

  if MY_TCODES:
    if(len(monthly_nonstationary) > 0):
      nonstationary_names, _ = zip(*monthly_nonstationary)

      monthly_params.loc[:, 'Tcode'] = update_Tcodes(monthly_nonstationary,
                                                monthly_params.loc[:,'Tcode'])

      monthly.loc[:, list(nonstationary_names)] = data_xform(
        monthly_original.dropna(),
        monthly_params.loc[:, 'Tcode'],
        xform_funcs,
        list(nonstationary_names)
      )

      pv = .05
      monthly_nonstationary = test_stationarity(monthly, adfuller_params, pv)
      print 'Number of nonstationary monthly series (my Tcodes) at p ~= %f: %d'\
        % (pv, len(monthly_nonstationary))

      pv = .01
      monthly_nonstationary = test_stationarity(monthly, adfuller_params, pv)
      print 'Number of nonstationary monthly series at (my Tcodes) p ~= %f: %d'\
        % (pv, len(monthly_nonstationary))

  #--------FIX QUARTERLY STATIONARITY--------
  #Test stationarity
  pv = .05
  quarterly_nonstationary = test_stationarity(quarterly, adfuller_params, pv)
  print 'Number of nonstationary quarterly series (SW Tcodes) at p ~= %f: %d'\
    % (pv, len(quarterly_nonstationary))

  pv = .01
  quarterly_nonstationary = test_stationarity(quarterly, adfuller_params, pv)
  print 'Number of nonstationary quarterly series (SW Tcodes) at p ~= %f: %d'\
    % (pv, len(quarterly_nonstationary))

  if MY_TCODES:
    if(len(quarterly_nonstationary)):
      quarterly_params.loc[:, 'Tcode'] = update_Tcodes(quarterly_nonstationary,
                                          quarterly_params.loc[:, 'Tcode'])
      nonstationary_names, _ = zip(*quarterly_nonstationary)

      quarterly.loc[:, list(nonstationary_names)] = data_xform(
        quarterly_original.dropna(),
        quarterly_params.loc[:, 'Tcode'],
        xform_funcs,
        list(nonstationary_names))

      pv = .05
      quarterly_nonstationary = test_stationarity(quarterly, adfuller_params, pv)
      print 'Number of nonstationary quarterly series (my Tcodes) at p ~= %f: %d'\
        % (pv, len(quarterly_nonstationary))

      pv = .01
      quarterly_nonstationary = test_stationarity(quarterly, adfuller_params, pv)
      print 'Number of nonstationary quarterly series (my Tcodes) at p ~= %f: %d'\
        % (pv, len(quarterly_nonstationary))

  #Sanity checks
  #This will produce a shitload of plots
  '''
  for s in monthly.columns:
    plt.plot(monthly[s])
    plt.title(s)
    plot_acf(monthly[s])
    plt.show()

  for s in quarterly.columns:
    plt.plot(quarterly[s])
    plt.title(s)
    plot_acf(quarterly[s])
    plt.show()
  '''

  #At least 41 of these time series seem (visually) to exhibit at least
  #some non-stationarity.  And, 13 have clear non-stationary characteristics.
  #Calling any of these series 'stationary' is probably pretty generous.
  
  #------WRITE TO AN OUTPUT FILE--------
  monthly.to_csv(SW_DIR + SW_MONTHLY_STATIONARY, **output_csv_params)
  quarterly.to_csv(SW_DIR + SW_QUARTERLY_STATIONARY, **output_csv_params)

if __name__ == '__main__':
  main()
