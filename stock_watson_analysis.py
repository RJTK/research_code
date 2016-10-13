import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from stock_watson_data import load_data

SW_DIR = '/home/ryan/Documents/academics/research/'\
         'granger_causality/software/datasets/stock_watson/data/'
SW_MONTHLY_STATIONARY = 'stock_watson_monthly_stationary.csv'
SW_QUARTERLY_STATIONARY = 'stock_watson_quarterly_stationary.csv'

iso8601_date_parser = lambda d : pd.datetime.strptime(d, '%Y-%m-%d')

csv_params = {'header' : 0,
              'index_col' : 0,
              'parse_dates' : True,
              'date_parser' : iso8601_date_parser
}

def main():
  monthly = load_data(SW_DIR + SW_MONTHLY_STATIONARY, csv_params)
  quarterly = load_data(SW_DIR + SW_QUARTERLY_STATIONARY, csv_params)
  
  return

if __name__ == '__main__':
  main()
