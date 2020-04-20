import os
import gc

import numpy as np
import pandas as pd
import lightgbm as lgb

from utils import timed

def valid_stats(preds, trues):
  assert preds.ndim == 1
  assert trues.ndim == 1
  assert len(preds) == len(trues)
  rows = len(preds)
  mse = np.mean((preds - trues)  * (preds - trues))
  mae = np.mean(np.abs(preds - trues))
  print(f'  MSE: {mse:.2f}, MAE: {mae:.2f}')

XY_CACHE = 'data/xy_cache.npz'
def load_data():
  if os.path.isfile(XY_CACHE):
    print('loading from cache')
    cached = np.load(XY_CACHE)
    return cached['X'], cached['y']

  with timed('loading data from csv...'):
    frame = pd.read_csv('data/sales_train_validation.csv')

    # input format:
    #   one row per item, one column per day
    items = 30490
    days = 1912
    sales = np.zeros((days, items), dtype=int)
    for d in range(days):
      sales[d, :] = frame[f'd_{d+1}'].values

    # y is (days, items) after the first year
    y = sales[365:,:]

    # start with a very simple timeseries prediction setup
    #
    # skip the first 365 days, then predict each day based on:
    #   the previous day
    #   the previous 7 days
    #   the previous 28 days
    #   the previous 365 days
    target_days = days - 365
    rows = items * target_days

    # ordinals for the categorical variables
    ordinals = {}
    for c, col in enumerate(['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']):
      _, inverse = np.unique(frame[col], return_inverse=True)
      ordinals[c] = inverse

    # this axes order is efficient for writing
    # approximate with uint16 for efficiency
    # scale floats before the uint16 truncation:
    #
    #   np.max(sales) == 763 so 80 is a safe scale for mean
    #   np.std is harder to predict, so use 10 to be more cautious
    feats = 19
    X = np.zeros((target_days, feats, items), dtype=np.uint16)
    for t in range(target_days):
      if (t % 100) == 0:
        print(f'{t}/{target_days}')
      d = t + 365

      # 1 feature: previous day
      X[t, 0] = sales[d-1]

      # 5 features: previous 7 days
      X[t, 1] = sales[d-7]
      X[t, 2] = np.mean(sales[d-7:d], axis=0) * 80
      X[t, 3] = np.min(sales[d-7:d], axis=0)
      X[t, 4] = np.max(sales[d-7:d], axis=0)
      X[t, 5] = np.std(sales[d-7:d], axis=0) * 10

      # 4 features: previous 28 days
      # (min is almost always going to be 0)
      X[t, 6] = sales[d-28]
      X[t, 7] = np.mean(sales[d-28:d], axis=0) * 80
      X[t, 8] = np.max(sales[d-28:d], axis=0)
      X[t, 9] = np.std(sales[d-28:d], axis=0) * 10

      # 5 features: previous 365 days
      # (min is almost always going to be 0)
      X[t, 10] = sales[d-365]
      X[t, 11] = np.mean(sales[d-365:d], axis=0) * 80
      X[t, 12] = np.max(sales[d-365:d], axis=0)
      X[t, 13] = np.std(sales[d-365:d], axis=0) * 10

      # 5 features: ordinals for the categorical variables
      X[t, 14] = ordinals[0]
      X[t, 15] = ordinals[1]
      X[t, 16] = ordinals[2]
      X[t, 17] = ordinals[3]
      X[t, 18] = ordinals[4]

  X = np.swapaxes(X, 1, 2)
  assert X.shape == (target_days, items, feats)

  with timed('saving...'):
    np.savez_compressed(XY_CACHE, X=X, y=y)

  return X, y


def split(X, y):
  days, items, feats = X.shape
  assert y.shape == (days, items)

  # cut off the last 3 months for validation sets
  # outputs are (rows, feats)
  valids = {
    'last month': (X[-28].reshape(-1, feats), y[-28].flatten()),
    '2nd to last month': (X[-56:-28].reshape(-1, feats), y[-56:-28].flatten()),
    '3rd to last month': (X[-84:-56].reshape(-1, feats), y[-84:-56].flatten()),
  }
  X = X[:-84]
  y = y[:-84]

  # for the remaining 1463 days, choose a random 20% to validate
  assert len(X) == 1463
  valid_count = len(X) // 5
  valid_idx = np.random.choice(np.arange(len(X)), size=valid_count, replace=False)
  is_valid = np.zeros(len(X), dtype=bool)
  is_valid[valid_idx] = True
  valids['random 20%'] = (X[is_valid].reshape(-1, feats), y[is_valid].flatten())
  train_X = X[~is_valid].reshape(-1, feats)
  train_y = y[~is_valid].flatten()
  return train_X, train_y, valids

def run():
  X, y = load_data()

  # TODO: don't include valid information in valid features
  # features need to come AFTER the day splits
  train_X, train_y, valids = split(X, y)
  del X
  del y
  gc.collect()

  with timed(f'training lightgbm with X.shape={train_X.shape}'):
    model = lgb.LGBMRegressor(n_estimators=10)
    model.fit(train_X, train_y)

  with timed('validating lightgbm...'):
    print('train:')
    valid_stats(model.predict(train_X), train_y)
    for valid_name, (valid_X, valid_y) in valids.items():
      print(f'{valid_name}:')
      valid_stats(model.predict(valid_X), valid_y)

if __name__ == '__main__':
  run()


