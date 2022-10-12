from gbdtmo import load_lib, GBDTSingle, GBDTMulti
import time, os, argparse
import numpy as np
import lightgbm as lgb
import cfg
from dataset import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("data")
parser.add_argument("-mode", default='gbdtmo', choices=['gbdtso', 'gbdtmo', 'lightgbm'])
parser.add_argument("-seed", default=0, type=int)
parser.add_argument("-k", default=0, type=int)
parser.add_argument("-N", default=10000, type=int)
args = parser.parse_args()

LIB = load_lib(cfg.Lib_path)
np.random.seed(args.seed)
ROUND = 1000
GAMMA = 1e-3
min_samples = 4 if args.data == 'yeast' else 16
num_threads = 4 if args.data in ['mnist', 'yeast'] else 8

if not os.path.isdir("log"): os.mkdir("log")


def train_lightgbm(data, meta):
    depth = cfg.Depth[args.mode][args.data]
    lr = cfg.Learning_rate[args.mode][args.data]

    p = {
        'boosting_type': 'gbdt',
        'objective': 'softmax',
        'metric': 'multi_error',
        'num_class': meta['out'],
        'num_leaves': int(0.75 * 2 ** depth),
        'learning_rate': lr,
        'num_threads': num_threads,
        'max_depth': depth,
        'lambda_l2': 1.0,
        'min_gain_to_split': GAMMA * 0.5,
        'min_data_in_leaf': min_samples,
        'max_bin': meta['bin'],
        'verbose': -1,
    }

    x_train, y_train, x_test, y_test = data
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
    m = lgb.train(p, lgb_train, num_boost_round=1000, valid_sets=lgb_eval,
                  early_stopping_rounds=25)

    preds = m.predict(x_test, num_iteration=m.best_iteration)
    preds = np.argmax(preds, -1)
    del m

    acc = (y_test == preds).sum()
    acc = acc / float(len(y_test))
    print("Best score: {:.5f}".format(acc))


def train_gbdt_multi(data, meta):
    depth = cfg.Depth[args.mode][args.data]
    lr = cfg.Learning_rate[args.mode][args.data]

    p = {'max_depth': depth, 'max_leaves': int(0.75 * 2 ** depth), 'topk': args.k, 'loss': b"ce",
         'gamma': GAMMA, 'num_threads': num_threads, 'max_bins': meta['bin'], 'lr': lr, 'reg_l2': 1.0,
         'early_stop': 25, 'one_side': True, 'verbose': True,
         'min_samples': min_samples}

    m = GBDTMulti(LIB, out_dim=meta['out'], params=p)
    x_train, y_train, x_test, y_test = data
    m.set_data((x_train, y_train), (x_test, y_test))
    m.train(ROUND)


def train_gbdt_single(data, meta):
    depth = cfg.Depth[args.mode][args.data]
    lr = cfg.Learning_rate[args.mode][args.data]

    p = {'max_depth': depth, 'max_leaves': int(0.75 * 2 ** depth), 'topk': 0, 'loss': b"ce_column",
         'gamma': GAMMA, 'num_threads': num_threads, 'max_bins': meta['bin'], 'lr': lr, 'reg_l2': 1.0,
         'early_stop': 25, 'one_side': True, 'verbose': True,
         'min_samples': min_samples}

    x_train, y_train, x_test, y_test = data

    m = GBDTSingle(LIB, out_dim=meta['out'], params=p)
    m.set_data((x_train, y_train), (x_test, y_test))
    m.train_multi(ROUND)


if __name__ == '__main__':
    m = DataLoader()
    data, meta = m.get(args.data, False, args.N)
    if args.mode == 'gbdtmo':
        train_gbdt_multi(data, meta)
    elif args.mode == 'gbdtso':
        train_gbdt_single(data, meta)
    elif args.mode == 'lightgbm':
        train_lightgbm(data, meta)
