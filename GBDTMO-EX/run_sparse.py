from gbdtmo import load_lib, GBDTMulti
import time, argparse
import numpy as np
import cfg
from dataset import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("data", help="which dataset to use",
                    choices=['mnist', 'mnist_reg', 'Caltech101', 'nus-wide'])
parser.add_argument("-time", help="whether to test running time", default=0, type=int)
parser.add_argument("-seed", default=0, type=int)
args = parser.parse_args()

LIB = load_lib(cfg.Lib_path)
np.random.seed(args.seed)
if args.time == 0:
    ROUND = 8000
else:
    ROUND = 10


def regression(data, meta, depth, lr, k, one_side):
    print("depth: {}, lr: {}, k: {}, one_side: {}".format(depth, lr, k, one_side))

    p = {'max_depth': depth, 'max_leaves': int(0.75 * 2 ** depth), 'topk': k, 'loss': b"mse",
         'gamma': 1e-6, 'num_threads': 8, 'max_bins': meta['bin'], 'lr': lr, 'reg_l2': 1.0,
         'early_stop': 25, 'one_side': one_side, 'verbose': False, 'hist_cache': 48,
         'min_samples': 4}

    m = GBDTMulti(LIB, out_dim=meta['out'], params=p)
    x_train, y_train, x_test, y_test = data
    m.set_data((x_train, y_train), (x_test, y_test))

    t = time.time()
    m.train(ROUND)
    t = time.time() - t
    if args.time == 1:
        print("Average time: {:.3f}".format(t/ROUND))
    else:
        print("Total time: {:.3f}".format(t))
    del m


def classification(data, meta, depth, lr, k, one_side):
    print("depth: {}, lr: {}, k: {}, one_side: {}".format(depth, lr, k, one_side))

    p = {'max_depth': depth, 'max_leaves': int(0.75 * 2 ** depth), 'topk': k, 'loss': b"ce",
         'gamma': 1e-3, 'num_threads': 8, 'max_bins': meta['bin'], 'lr': lr, 'reg_l2': 1.0,
         'early_stop': 25, 'one_side': one_side, 'verbose': False,
         'min_samples': 16}

    m = GBDTMulti(LIB, out_dim=meta['out'], params=p)
    x_train, y_train, x_test, y_test = data
    m.set_data((x_train, y_train), (x_test, y_test))
    t = time.time()
    m.train(ROUND)
    t = time.time() - t
    if args.time == 1:
        print("Average time: {:.3f}".format(t/ROUND))
    else:
        print("Total time: {:.3f}".format(t))
    del m


if __name__ == '__main__':
    m = DataLoader()
    if args.data == 'mnist_reg':
        data, meta = m.get('mnist_reg', True, 10000)
        for k in [4, 8, 16]:
            regression(data, meta, 7, 0.1, k, True)
            regression(data, meta, 7, 0.1, k, False)
    elif args.data == 'nus-wide':
        data, meta = m.get('nus-wide', True, 10000)
        for k in [8, 16, 32, 64]:
            regression(data, meta, 8, 0.1, k, True)
            regression(data, meta, 8, 0.1, k, False)
    elif args.data == 'Caltech101':
        data, meta = m.get('Caltech101', False, 10000)
        for k in [8, 16, 32, 64]:
            classification(data, meta, 8, 0.1, k, True)
            classification(data, meta, 8, 0.1, k, False)
    elif args.data == 'mnist':
        data, meta = m.get('mnist', False, 10000)
        for k in [2, 4, 8]:
            classification(data, meta, 8, 0.1, k, True)
            classification(data, meta, 8, 0.1, k, False)
    else:
        raise ValueError("Unknown dataset!")
    
