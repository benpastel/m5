import os
import gc

import numpy as np
import pandas as pd
import lightgbm as lgb

from utils import timed

import matplotlib.pyplot as plt

ALL_DAYS = 1969
DATA_DAYS = 1913
ALL_IDS = 30490
VALID_DAYS = 28
SKIP_DAYS = 365 + VALID_DAYS

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
  sales = np.zeros((DATA_DAYS, ALL_IDS), dtype=int)
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

    # y is (days, ids) after the first year
    y = sales[SKIP_DAYS:,:]

    uniques = {}
    ordinals = {}
    for col in ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']:
      u, inverse = np.unique(sales_frame[col], return_inverse=True)
      uniques[col] = u
      ordinals[col] = inverse

    target_days = DATA_DAYS - SKIP_DAYS
    feats = 5 + 4 + (5 * 2)
    X = np.zeros((target_days, feats, ALL_IDS), dtype=np.float32)

    # broadcast over all ids
    train_day_feats = day_feats[SKIP_DAYS:SKIP_DAYS+target_days,:]
    X[:, 0:5, :] = train_day_feats.reshape(target_days, 5, 1)

    # for dept_id, cat_id, store_id, state_id there are few enough values we can just
    # pass them in as ordinals directly.
    # broadcast over all days
    X[:, 5, :] = ordinals['dept_id']
    X[:, 6, :] = ordinals['cat_id']
    X[:, 7, :] = ordinals['store_id']
    X[:, 8, :] = ordinals['state_id']

    # for id and item_id, try a bunch of different embeddings.
    # for now, all sales aggregations must be validation-safe:
    # they only use info from t - VALID_DAYS and earlier.

    # start with:
    #   (1) mean, min, max, std, nonzero over previous year
    # TODO:
    #   try over different windows (month, week)
    #   try over different slices (sharing holiday; sharing day of week; etc.)

    # ID features
    for t in range(target_days):
      if (t % 100) == 0:
        print(f'{t}/{target_days}')

      d = SKIP_DAYS + t - VALID_DAYS

      assert d-365 >= 0
      group = sales[d-365:d]

      X[t, 9, :] = np.mean(group, axis=0)
      X[t, 10, :] = np.min(group, axis=0)
      X[t, 11, :] = np.max(group, axis=0)
      X[t, 12, :] = np.std(group, axis=0)
      X[t, 13, :] = np.count_nonzero(group, axis=0)

      for i in range(len(uniques['item_id'])):
        is_item = (ordinals['item_id'] == i)
        item_group = group[:, is_item]

        # print(f'{ordinals["item_id"].shape=}')
        # print(f'{is_item.shape=}, {np.count_nonzero(is_item)=}')
        # print(f'{group.shape=}, {item_group.shape=}, {np.mean(item_group, axis=0).shape=}, {X[t, 14, is_item].shape=}')

        X[t, 14, is_item] = np.mean(item_group, axis=0)
        X[t, 15, is_item] = np.min(item_group, axis=0)
        X[t, 16, is_item] = np.max(item_group, axis=0)
        X[t, 17, is_item] = np.std(item_group, axis=0)
        X[t, 18, is_item] = np.count_nonzero(item_group, axis=0)

  X = np.swapaxes(X, 1, 2)
  assert X.shape == (target_days, ALL_IDS, feats)

  with timed('saving...'):
    np.savez_compressed(XY_CACHE, X=X, y=y)
  return X, y


def validate_on_end():
  X, y = load_data()
  days, ids, feats = X.shape
  assert y.shape == (days, ids)
  assert ids == ALL_IDS
  assert days == DATA_DAYS - SKIP_DAYS

  valid_X = X[-VALID_DAYS:].reshape(-1, feats)
  valid_y = y[-VALID_DAYS:].flatten()
  train_X = X[:-VALID_DAYS].reshape(-1, feats)
  train_y = y[:-VALID_DAYS].flatten()
  del X
  del y
  gc.collect()

  with timed(f'training lightgbm with X.shape={train_X.shape}'):
    model = lgb.LGBMRegressor(n_estimators=100)
    model.fit(train_X, train_y)

  print('train error:')
  valid_stats(model.predict(train_X), train_y, should_print=True)

  print('valid error:')
  valid_stats(model.predict(valid_X), valid_y, should_print=True)


if __name__ == '__main__':
  validate_on_end()


