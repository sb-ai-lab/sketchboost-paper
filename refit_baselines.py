import argparse

import joblib
import os

from itertools import product
from joblib import Parallel, delayed
from multiprocessing import Queue

from utils import run_cv_loop

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--ngpus', type=int)

NAME = 'BASELINES_REFIT'

RUNNER_PATH = 'runner.py'
DEBUG = True

TIMEOUT = 3600 * 24
HARD_TIMEOUT = 3600 * 36
NTHREADS = 8
SEED = 42

# benchmark_params
data_path = 'data/processed'
benchmark_path = 'runs'


def get_baseline(name, benchmark_path, data_path, dataset, task, rewrite=False):
    gpu = q.get()

    params = joblib.load(os.path.join(benchmark_path, 'baselines_and_params.pkl'))[dataset, task]['params']
    params = {**params, **{'lr': 0.015, 'ntrees': 20000, 'es': 500}}

    run_cv_loop(name, gpu, benchmark_path, data_path, dataset, task, 'default', params, rewrite=rewrite)

    q.put(gpu)


if __name__ == '__main__':

    args = parser.parse_args()

    q = Queue(maxsize=args.ngpus)
    for i in range(args.ngpus):
        q.put([i])

    data_info = joblib.load(os.path.join(data_path, 'data_info.pkl'))
    datasets = ['otto', 'dionis', 'helena', 'sf-crime', 'moa', 'scm20d', 'rf1', 'delicious', 'mediamill', ]

    tasks = product(datasets, ['cb', 'xgb'])

    Parallel(n_jobs=args.ngpus, backend="threading")(
        delayed(get_baseline)(NAME, benchmark_path, data_path, ds, fr, rewrite=False) for (ds, fr) in tasks
    )
