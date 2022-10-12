from gbdtmo import load_lib, GBDTSingle, GBDTMulti
import time, copy, argparse
import numpy as np
import cfg
from dataset import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("data", help="which dataset to use",
                    choices=['mnist', 'mnist_reg', 'Caltech101', 'nus-wide'])
parser.add_argument("-workers", default=8, type=int)
args = parser.parse_args()

LIB = load_lib(cfg.Lib_path)
ROUND = 10


def classification_multi(data, meta, depth, lr):
    p = {'max_depth': depth, 'max_leaves': int(0.75 * 2 ** depth), 'topk': 0, 'loss': b"ce",
         'gamma': 1e-3, 'num_threads': args.workers, 'max_bins': meta['bin'], 'lr': lr, 'reg_l2': 1.0,
         'early_stop': 0, 'one_side': True, 'verbose': False,
         'min_samples': 16}

    m = GBDTMulti(LIB, out_dim=meta['out'], params=p)
    x_train, y_train, x_test, y_test = data
    m.set_data((x_train, y_train))
    t = time.time()
    m.train(ROUND)
    t = time.time() - t
    del m
    return t


def classification_single(data, meta, depth, lr):
    p = {'max_depth': depth, 'max_leaves': int(0.75 * 2 ** depth), 'topk': 0, 'loss': b"ce_column",
         'gamma': 1e-3, 'num_threads': args.workers, 'max_bins': meta['bin'], 'lr': lr, 'reg_l2': 1.0,
         'early_stop': 0, 'one_side': True, 'verbose': False,
         'min_samples': 16}

    x_train, y_train, x_test, y_test = data

    m = GBDTSingle(LIB, out_dim=meta['out'], params=p)
    m.set_data((x_train, y_train))
    t = time.time()
    m.train_multi(ROUND)
    t = time.time() - t
    del m
    return t


def regression_multi(data, meta, depth, lr):
    p = {'max_depth': depth, 'max_leaves': int(0.75 * 2 ** depth), 'topk': 0, 'loss': b"mse",
         'gamma': 1e-6, 'num_threads': args.workers, 'max_bins': meta['bin'], 'lr': lr, 'reg_l2': 1.0,
         'early_stop': 0, 'one_side': True, 'verbose': False, 'hist_cache': 48,
         'min_samples': 4}

    m = GBDTMulti(LIB, out_dim=meta['out'], params=p)
    x_train, y_train, x_test, y_test = data
    m.set_data((x_train, y_train))
    t = time.time()
    m.train(ROUND)
    t = time.time() - t
    del m
    return t


def regression_single(data, meta, depth, lr):
    p = {'max_depth': depth, 'max_leaves': int(0.75 * 2 ** depth), 'topk': 0, 'loss': b"mse",
         'gamma': 1e-6, 'num_threads': args.workers, 'max_bins': meta['bin'], 'lr': lr, 'reg_l2': 1.0,
         'early_stop': 0, 'one_side': True, 'verbose': False, 'hist_cache': 48,
         'min_samples': 4}

    x_train, y_train, x_test, y_test = data
    m = GBDTSingle(LIB, out_dim=1, params=p)
    m.set_data((x_train, None))

    t = 0
    for i in range(meta['out']):
        yy_train = np.ascontiguousarray(y_train[:, i])
        m._set_label(yy_train, True)
        _t = time.time()
        m.train(ROUND)
        t += time.time() - _t
        m.reset()
    del m
    return t


if __name__ == '__main__':
    p = {'mnist': [8, 0.1],
         'Caltech101': [9, 0.1],
         'mnist_reg': [7, 0.1],
         'nus-wide': [8, 0.1],
         }

    name = args.data
    m = DataLoader()
    out_multi, out_single = 0.0, 0.0
    for i in range(3):
        if name == 'mnist':
            data, meta = m.get(name, False, 10000)
            out_multi += classification_multi(data, meta, p[name][0], p[name][1])
            out_single += classification_single(data, meta, p[name][0], p[name][1])
        elif name == 'Caltech101':
            data, meta = m.get(name, False, 10000)
            out_multi += classification_multi(data, meta, p[name][0], p[name][1])
            out_single += classification_single(data, meta, p[name][0], p[name][1])
        elif name == 'mnist_reg':
            data, meta = m.get(name, True, 10000)
            out_multi += regression_multi(data, meta, p[name][0], p[name][1])
            out_single += regression_single(data, meta, p[name][0], p[name][1])
        elif name == 'nus-wide':
            data, meta = m.get(name, True, args.N)
            out_multi += regression_multi(data, meta, p[name][0], p[name][1])
            out_single += regression_single(data, meta, p[name][0], p[name][1])

    out_single /= 3.0 * ROUND
    out_multi /= 3.0 * ROUND

    print("GBDT-SO on {}:\t{:.3f}".format(name, out_single))
    print("GBDT-MO on {}:\t{:.3f}".format(name, out_multi))







