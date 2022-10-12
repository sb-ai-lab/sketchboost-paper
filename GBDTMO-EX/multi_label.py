from gbdtmo import load_lib, GBDTSingle, GBDTMulti
import time, os, argparse
import numpy as np
import lightgbm as lgb
import cfg
from dataset import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("data")
parser.add_argument("-mode", default='gbdtmo', choices=['gbdtmo', 'gbdtso', 'lightgbm'])
parser.add_argument("-seed", default=0, type=int)
parser.add_argument("-N", default=10000, type=int)
args = parser.parse_args()

LIB = load_lib(cfg.Lib_path)
np.random.seed(args.seed)
ROUND = 8000
GAMMA = 1e-6

if not os.path.isdir("log"): os.mkdir("log")
if not os.path.isdir("result"): os.mkdir("result")


def train_lightgbm(data, meta):
    depth = cfg.Depth[args.mode][args.data]
    lr = cfg.Learning_rate[args.mode][args.data]

    p = {
        'boosting_type': 'gbdt',
        'objective': 'mse',
        'num_leaves': int(0.75 * 2 ** depth),
        'learning_rate': lr,
        'num_threads': 8,
        'max_depth': depth,
        'lambda_l2': 1.0,
        'min_gain_to_split': GAMMA * 0.5,
        'min_data_in_leaf': 8,
        'max_bin': meta['bin'],
        'verbose': -1,
    }

    x_train, y_train, x_test, y_test = data
    preds = np.zeros((len(y_test), meta['out']), 'float64')
    lgb_train = lgb.Dataset(x_train)
    lgb_train.construct()
    lgb_eval = lgb.Dataset(x_test)
    
    for i in range(meta['out']):
        yy_train = np.ascontiguousarray(y_train[:, i])
        yy_test = np.ascontiguousarray(y_test[:, i])        
        lgb_train.set_label(yy_train)
        lgb_eval.set_label(yy_test)
        m = lgb.train(p, lgb_train, num_boost_round=ROUND, valid_sets=lgb_eval,
                      early_stopping_rounds=25)
        _ = m.predict(x_test, num_iteration=m.best_iteration)
        preds[:, i] = _
        del m
        
    np.save("result/lightgbm", preds)


def train_gbdt_multi(data, meta):
    depth = cfg.Depth[args.mode][args.data]
    lr = cfg.Learning_rate[args.mode][args.data]

    p = {'max_depth': depth, 'max_leaves': int(0.75 * 2 ** depth), 'topk': 0, 'loss': b"mse",
         'gamma': GAMMA, 'num_threads': 8, 'max_bins': meta['bin'], 'lr': lr, 'reg_l2': 1.0,
         'early_stop': 25, 'one_side': True, 'verbose': False, 'hist_cache': 48,
         'min_samples': 8}

    m = GBDTMulti(LIB, out_dim=meta['out'], params=p)
    x_train, y_train, x_test, y_test = data
    m.set_data((x_train, y_train), (x_test, y_test))
    m.train(ROUND)
    preds = m.predict(x_test)
    del m
    np.save("result/gbdtm", preds)


def train_gbdt_single(data, meta):
    depth = cfg.Depth[args.mode][args.data]
    lr = cfg.Learning_rate[args.mode][args.data]

    p = {'max_depth': depth, 'max_leaves': int(0.75 * 2 ** depth), 'topk': 0, 'loss': b"mse",
         'gamma': GAMMA, 'num_threads': 8, 'max_bins': meta['bin'], 'lr': lr, 'reg_l2': 1.0,
         'early_stop': 25, 'one_side': True, 'verbose': False, 'hist_cache': 48,
         'min_samples': 8}

     x_train, y_train, x_test, y_test = data
    m = GBDTSingle(LIB, out_dim=1, params=p)
    preds = np.zeros_like(y_test)
    m.set_data((x_train, None), (x_test, None))
    
    for i in range(meta['out']):
        yy_train = np.ascontiguousarray(y_train[:, i])
        yy_test = np.ascontiguousarray(y_test[:, i])
        m._set_label(yy_train, True)
        m._set_label(yy_test, False)
        m.train(ROUND)
        _ = m.predict(x_test)
        preds[:, i] = _
        m.reset()
    np.save("result/gbdt", preds)


if __name__ == '__main__':
    m = DataLoader()
    data, meta = m.get(args.data, False, args.N)

    if args.mode == 'gbdtmo':
        train_gbdt_multi(data, meta)
    elif args.mode == 'gbdtso':
        train_gbdt_single(data, meta)
    elif args.mode == 'lightgbm':
        train_lightgbm(data, meta)
