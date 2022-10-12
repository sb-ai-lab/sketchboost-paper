import argparse

import joblib
import os

from itertools import product
from joblib import Parallel, delayed
from multiprocessing import Queue

from utils import run_cv_loop

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--ngpus', type=int)

NAME = 'GBDTMO_BESTK'

RUNNER_PATH = 'runner.py'
DEBUG = True

TIMEOUT = 3600 * 24
HARD_TIMEOUT = 3600 * 36
NTHREADS = 8
SEED = 42

# benchmark_params
data_path = 'data/processed'
benchmark_path = 'runs'


def get_pb_results(name, benchmark_path, data_path, dataset, params, method, dim, rewrite=False):
    gpu = q.get()

    trial = method + '_' + str(dim)

    if method == 'randomhess':
        method = 'random'
        use_hess = True
    else:
        use_hess = False

    params['method'] = method
    params['dim'] = dim
    params['use_hess'] = use_hess

    print(params)

    run_cv_loop(name, gpu, benchmark_path, data_path, dataset, 'pb', trial, params, rewrite=rewrite)

    q.put(gpu)


params = {

    'caltech': {
        'lr': 0.1,
        'max_bin': 32 + 1,
        'max_depth': 10,
        'min_data_in_leaf': 16,
        'lambda_l2': 1,
        'es': 25,
        'ntrees': 8000,
        'cpu': True,
        'subsample': 1,
        'acc': True,

    },
    'nuswide': {
        'lr': 0.1,
        'max_bin': 64 + 1,
        'max_depth': 8,
        'min_data_in_leaf': 4,
        'lambda_l2': 1,
        'es': 25,
        'ntrees': 8000,
        'cpu': True,
        'subsample': 1,
        'acc': True,

    },

    'mnist': {
        'lr': 0.1,
        'max_bin': 8 + 1,
        'max_depth': 8,
        'min_data_in_leaf': 16,
        'lambda_l2': 1,
        'es': 25,
        'ntrees': 8000,
        'cpu': True,
        'subsample': 1,
        'acc': True,

    },

    'mnistreg': {
        'lr': 0.1,
        'max_bin': 16 + 1,
        'max_depth': 7,
        'min_data_in_leaf': 4,
        'lambda_l2': 1,
        'es': 25,
        'ntrees': 8000,
        'cpu': True,
        'subsample': 1,
        'acc': True,

    }

}

if __name__ == '__main__':

    args = parser.parse_args()

    q = Queue(maxsize=args.ngpus)
    for i in range(args.ngpus):
        q.put([i])

    data_info = joblib.load(os.path.join(data_path, 'data_info.pkl'))
    datasets = ['caltech', 'nuswide', 'mnist', 'mnistreg']
    strats = ['random', 'proj', ]
    ks = [1, 2, 5, 10, 20]

    combinations = [(ds, method, k) for (ds, method, k) in product(datasets, strats, ks) if data_info[ds]['nout'] > k
                    ] + [(ds, 'raw', 10000) for ds in datasets]

    # get runs

    Parallel(n_jobs=args.ngpus, backend="threading")(
        delayed(get_pb_results)(
            NAME, benchmark_path, data_path, d, params[d], p, dim, rewrite=False) for d, p, dim in combinations
    )
