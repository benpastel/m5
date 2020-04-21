import os
import gc

import numpy as np
import pandas as pd
import lightgbm as lgb

from utils import timed

import matplotlib.pyplot as plt

ALL_DAYS = 1969
DATA_DAYS = 1913
ALL_ITEMS = 30490
SKIP_DAYS = 365
VALID_DAYS = 84

def valid_stats(preds, trues, should_print=True):
  assert preds.ndim == 1
  assert trues.ndim == 1
  assert len(preds) == len(trues)
  rows = len(preds)
  mse = np.mean((preds - trues)  * (preds - trues))
  mae = np.mean(np.abs(preds - trues))
  if should_print:
    print(f'  MSE: {mse:.2f}, MAE: {mae:.2f}')
  return mse, mae


def load_sales():
  frame = pd.read_csv('data/sales_train_validation.csv')
  sales = np.zeros((DATA_DAYS, ALL_ITEMS), dtype=int)
  for d in range(DATA_DAYS):
    sales[d, :] = frame[f'd_{d+1}'].values

  return sales




XY_CACHE = 'data/xy_cache.npz'
def load_data():
  if os.path.isfile(XY_CACHE):
    print('loading from cache')
    cached = np.load(XY_CACHE)
    return cached['X'], cached['y']

  with timed('loading data from csv...'):
    day_frame = pd.read_csv('data/calendar.csv')

    # feats:
    #   day of week
    #   day of month
    #   month
    #   event_1 (encoded in [0, 32])
    #   snap    (encoded in [0, 2**3])
    day_feats = np.zeros((ALL_DAYS, 5), dtype=np.uint8)
    assert len(day_frame) == ALL_DAYS
    day_feats[:, 0] = day_frame['wday'].values

    # parse the day part from YYYY-MM-DD
    day_feats[:, 1] = [int(date.split('-')[2]) for date in day_frame['date']]

    day_feats[:, 2] = day_frame['month'].values

    # for simplicity, ignore event_name_2
    # TODO try using it
    _, event_codes = np.unique(day_frame['event_name_1'].values.astype(str), return_inverse=True)
    day_feats[:, 3] = event_codes

    # for simplicity, dense-code the snap
    # TODO eventually match it properly based on geography
    day_feats[:, 4] = (
      day_frame['snap_CA'].values +
      2 * day_frame['snap_TX'].values +
      4 * day_frame['snap_WI'].values)

    assert np.min(day_feats[:, 0]) == 1
    assert np.max(day_feats[:, 0]) == 7
    assert np.min(day_feats[:, 1]) == 1
    assert np.max(day_feats[:, 1]) == 31
    assert np.min(day_feats[:, 2]) == 1
    assert np.max(day_feats[:, 2]) == 12
    assert np.min(day_feats[:, 3]) == 0
    assert np.max(day_feats[:, 3]) == 30
    assert np.min(day_feats[:, 4]) == 0
    assert np.max(day_feats[:, 4]) == 7

    sales_frame = pd.read_csv('data/sales_train_validation.csv')
    sales = load_sales()

    # y is (days, items) after the first year
    y = sales[SKIP_DAYS:,:]

    # start with a very simple timeseries prediction setup
    #
    # skip the first 365 days, then predict each day based on:
    #   the previous day
    #   the previous 7 days
    #   the previous 28 days
    #   the previous 365 days
    target_days = DATA_DAYS - SKIP_DAYS
    rows = ALL_ITEMS * target_days

    ordinals = {}
    for c, col in enumerate(['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']):
      _, inverse = np.unique(sales_frame[col], return_inverse=True)
      ordinals[c] = inverse

    # this axes order is efficient for writing
    # approximate with uint16 for efficiency
    # scale floats before the uint16 truncation:
    #
    #   np.max(sales) == 763 so 80 is a safe scale for mean
    #   np.std is harder to predict, so use 10 to be more cautious
    feats = 29
    X = np.zeros((target_days, feats, ALL_ITEMS), dtype=np.uint16)
    for t in range(target_days):
      if (t % 100) == 0:
        print(f'{t}/{target_days}')
      d = t + SKIP_DAYS

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

      # broadcast the 5 day features across all items
      X[t, 19:24, :] = day_feats[d].reshape(-1, 1)

  X = np.swapaxes(X, 1, 2)
  assert X.shape == (target_days, ALL_ITEMS, feats)

  with timed('saving...'):
    np.savez_compressed(XY_CACHE, X=X, y=y)
  return X, y


def validate_on_end():
  X, y = load_data()
  days, items, feats = X.shape
  assert y.shape == (days, items)

  # cut off the last month for validation set
  # but we're going to re-calc the features as we go based on predictions
  oracle_valid_X = X[-VALID_DAYS:]
  valid_y = y[-VALID_DAYS:]
  train_X = X[:-VALID_DAYS].reshape(-1, feats)
  train_y = y[:-VALID_DAYS].flatten()

  assert valid_y.shape == (VALID_DAYS, items)
  assert len(train_X) == len(train_y)
  del X
  del y
  gc.collect()

  with timed(f'training lightgbm with X.shape={train_X.shape}'):
    model = lgb.LGBMRegressor(n_estimators=100)
    model.fit(train_X, train_y)

  print('train error:')
  valid_stats(model.predict(train_X), train_y, should_print=True)

  with timed('validating lightgbm...'):
    daily_mses = np.zeros(VALID_DAYS)
    daily_maes = np.zeros(VALID_DAYS)

    sales = load_sales()

    valid_preds = np.zeros((VALID_DAYS, ALL_ITEMS), dtype=np.float32)
    for t in range(VALID_DAYS):
      # print(f'day {t}:')
      X = np.zeros((feats, ALL_ITEMS), dtype=np.uint8)

      # t is the index into valid_preds
      # d is the index into sales
      d = len(sales) - VALID_DAYS + t

      # 1 feature: previous day
      X[0] = sales[d-1]

      # 5 features: previous 7 days
      X[1] = sales[d-7]
      X[2] = np.mean(sales[d-7:d], axis=0) * 80
      X[3] = np.min(sales[d-7:d], axis=0)
      X[4] = np.max(sales[d-7:d], axis=0)
      X[5] = np.std(sales[d-7:d], axis=0) * 10

      # 4 features: previous 28 days
      # (min is almost always going to be 0)
      X[6] = sales[d-28]
      X[7] = np.mean(sales[d-28:d], axis=0) * 80
      X[8] = np.max(sales[d-28:d], axis=0)
      X[9] = np.std(sales[d-28:d], axis=0) * 10

      # 5 features: previous 365 days
      # (min is almost always going to be 0)
      X[10] = sales[d-365]
      X[11] = np.mean(sales[d-365:d], axis=0) * 80
      X[12] = np.max(sales[d-365:d], axis=0)
      X[13] = np.std(sales[d-365:d], axis=0) * 10

      # 5 ID + 5 day features don't depend on future information
      # these don't depend on future information so we can use the oracle features directly
      X[14:24, :] = oracle_valid_X[t, :, 14:24].T

      # (feats, items) => (items, feats)
      X = X.T
      day_preds = model.predict(X)
      assert day_preds.shape == (ALL_ITEMS,)
      valid_preds[t] = day_preds

      # overwrite sales with predicted sale on this day
      sales[d, :] = day_preds

      mse, mae = valid_stats(day_preds, valid_y[t], should_print=False)
      daily_mses[t] = mse
      daily_maes[t] = mae

  print(f'validation on last {VALID_DAYS}:')
  valid_stats(valid_preds.flatten(), valid_y.flatten(), should_print=True)

  idx = np.arange(VALID_DAYS)
  # plt.plot(idx, daily_mses, 'bo', idx, daily_maes, 'r+')
  plt.plot(idx, daily_maes, 'r+')
  plt.show()


if __name__ == '__main__':
  validate_on_end()


