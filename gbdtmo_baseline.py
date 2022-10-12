import argparse

from copy import deepcopy
from joblib import Parallel, delayed
from multiprocessing import Queue

from utils import run_cv_loop

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--njobs', type=int)

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


def get_baseline_gbdtmo(name, benchmark_path, data_path, dataset, task, dim, one_side, params, rewrite=False):
    gpu = q.get()
    params = deepcopy(params)
    params['dim'] = dim
    params['one_side'] = one_side

    method = 'raw' if dim == 0 else 'sparse'
    trial = method + '_' + str(dim) + '_' + str(one_side)

    run_cv_loop(name, gpu, benchmark_path, data_path, dataset, task, trial, params, rewrite=rewrite)

    q.put(gpu)


params = [

    # caltech
    {
        'lr': 0.1,
        'max_bin': 32,
        'max_depth': 10,
        'min_data_in_leaf': 16,
        'lambda_l2': 1,
        'min_gain_to_split': 1e-3,

        'es': 25,
        'ntrees': 8000,
        'cpu': True,
        'subsample': 1,

    },
    # nuswide
    {
        'lr': 0.1,
        'max_bin': 64,
        'max_depth': 8,
        'min_data_in_leaf': 4,
        'lambda_l2': 1,
        'min_gain_to_split': 1e-6,

        'es': 25,
        'ntrees': 8000,
        'cpu': True,
        'subsample': 1,

    },

    # mnist
    {
        'lr': 0.1,
        'max_bin': 8,
        'max_depth': 8,
        'min_data_in_leaf': 16,
        'lambda_l2': 1,
        'min_gain_to_split': 1e-3,

        'es': 25,
        'ntrees': 8000,
        'cpu': True,
        'subsample': 1,

    },
    # mnist reg
    {
        'lr': 0.1,
        'max_bin': 16,
        'max_depth': 7,
        'min_data_in_leaf': 4,
        'lambda_l2': 1,
        'min_gain_to_split': 1e-6,

        'es': 25,
        'ntrees': 8000,
        'cpu': True,
        'subsample': 1,

    }

]

kvals = [

    [0, 64],
    [0, 16],
    [0, 4],
    [0, 16],

]

one_sides = [True, False, True, True]

if __name__ == '__main__':

    args = parser.parse_args()

    q = Queue(maxsize=args.njobs)
    for i in range(args.njobs):
        q.put([i])

    params_set = []

    for n, dat in enumerate(['caltech', 'nuswide', 'mnist', 'mnistreg']):

        p = params[n]
        oside = one_sides[n]

        for k in kvals[n]:
            params_set.append((dat, p, k, oside))

    # get runs

    Parallel(n_jobs=args.njobs, backend="threading")(
        delayed(get_baseline_gbdtmo)(NAME,
                                     benchmark_path,
                                     data_path,
                                     dat,
                                     'gbdtmo',
                                     d, oside,
                                     p,
                                     rewrite=True
                                     ) for (dat, p, d, oside) in params_set

    )
