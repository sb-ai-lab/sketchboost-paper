import argparse

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


def get_baseline(name, benchmark_path, data_path, dataset, task, params, rewrite=False):
    gpu = q.get()
    run_cv_loop(name, gpu, benchmark_path, data_path, dataset, task, 'default', params, rewrite=rewrite)

    q.put(gpu)


params = [

    # caltech
    {
        'lr': 0.1,
        'max_bin': 32,
        'max_depth': 10,
        'min_data_in_leaf': 16,
        'lambda_l2': 1,
        'es': 25,
        'ntrees': 8000,
        'cpu': True,
        'subsample': 1,
        'acc': True,

    },
    # nuswide
    {
        'lr': 0.1,
        'max_bin': 64,
        'max_depth': 8,
        'min_data_in_leaf': 4,
        'lambda_l2': 1,
        'es': 25,
        'ntrees': 8000,
        'cpu': True,
        'subsample': 1,
        'acc': True,

    },
    # mnist
    {
        'lr': 0.1,
        'max_bin': 8,
        'max_depth': 8,
        'min_data_in_leaf': 16,
        'lambda_l2': 1,
        'es': 25,
        'ntrees': 8000,
        'cpu': True,
        'subsample': 1,
        'acc': True,

    },
    # mnist reg
    {
        'lr': 0.1,
        'max_bin': 16,
        'max_depth': 7,
        'min_data_in_leaf': 4,
        'lambda_l2': 1,
        'es': 25,
        'ntrees': 8000,
        'cpu': True,
        'subsample': 1,
        'acc': True,

    }

]

if __name__ == '__main__':

    args = parser.parse_args()

    q = Queue(maxsize=args.njobs)
    for i in range(args.njobs):
        q.put([i])

    # get runs 
    Parallel(n_jobs=args.njobs, backend="threading")(
        delayed(get_baseline)(
            NAME, benchmark_path, data_path, d, 'cb', p, rewrite=True) for (d, p) in
        zip(['caltech', 'nuswide', 'mnist', 'mnistreg'], params)

    )
