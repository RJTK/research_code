import pyedflib
import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm

from matplotlib import pyplot as plt
from pandas.tools.plotting import autocorrelation_plot
from scipy import signal as sps

def read_data(T = 10):
  '''
  T is the number of seconds of data to collect
  '''
  #CARE: EdfReader keeps the file open
  edf_reader = pyedflib.EdfReader('datasets/eeg_data.edf')
  f = edf_reader.samplefrequency(0) #assume f is constant accross signals
  n = edf_reader.signals_in_file #Number of signals

  eeg = {edf_reader.signal_label(i).strip() : 
         edf_reader.readSignal(i)[:int(f*T)]
         for i in range(n)}
  eeg = pd.DataFrame(eeg)
  edf_reader._close()

  #For some stupid reason, VAR doesn't accept indices unless they are dates
  eeg.index = sm.tsa.datetools.dates_from_range('2016m1', None, len(eeg))

  return eeg

def fit_var(eeg, num_diffs = 0, p = 10):
  '''
  Returns a statsmodels VAR instance from eeg data returned by read_data(1)

  eeg_instance - eeg data from read_data
  num_diffs - the number of times to difference the data
  '''
  for i in range(num_diffs):
    eeg = eeg.diff('columns').dropna()

  var = sm.tsa.VAR(eeg)
  result = var.fit(p)
  return result

def main():
  '''
  '''
  eeg = read_data(T = 5)
  result = fit_var(eeg)
  return

if __name__ == '__main__':
  main()
