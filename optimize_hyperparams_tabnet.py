import argparse
import os
import subprocess
import sys
from itertools import product
from multiprocessing import Queue

import joblib
import numpy as np
from joblib import Parallel, delayed

from utils import run_cv_loop

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--ngpus', type=int)
parser.add_argument('-m', '--mode', type=str, default='run')

NAME = 'BASELINES_AND_PARAMS'

RUNNER_PATH = 'runner.py'
DEBUG = True

TIMEOUT = 3600 * 24
HARD_TIMEOUT = 3600 * 36
NTHREADS = 8
SEED = 42

# benchmark_params
data_path = 'data/processed'
benchmark_path = 'runs'


def get_baseline_and_tune(name, benchmark_path, data_path, dataset, task, rewrite=False):
    gpu = q.get()
    if task == 'tabnet':
        default_params = {'lr': 2e-2}
    else:
        default_params = {
            'max_depth': 6,
            'min_data_in_leaf': 1 if task == 'cb' else 1e-5,
            'lambda_l2': 1,
            'subsample': 1.,

        }

    run_cv_loop(name, gpu, benchmark_path, data_path, dataset, task, 'default', default_params, rewrite=rewrite)

    try:
        log = subprocess.check_output(' '.join([
            sys.executable,
            'params_search.py',
            '-b', benchmark_path,
            '-p', data_path,
            '-k', dataset,
            '-n', NAME,
            '-d', ','.join(map(str, gpu)),
            '-r', task

        ]), timeout=HARD_TIMEOUT, shell=True, stderr=subprocess.STDOUT, executable='/bin/bash').decode()

    except subprocess.TimeoutExpired:

        pass

    q.put(gpu)


def get_result(folder):
    files = os.listdir(folder)

    val_scores, test_scores = [], []

    for f in files:

        fname = os.path.join(folder, f, 'results.pkl')
        try:
            res = joblib.load(fname)
        except FileNotFoundError:
            continue

        val_scores.append(res['val_score'])
        test_scores.append(res['test_score'])

    valid = len(val_scores) == 5

    return np.mean(val_scores), np.mean(test_scores), valid


if __name__ == '__main__':

    args = parser.parse_args()

    q = Queue(maxsize=args.ngpus)
    for i in range(args.ngpus):
        q.put([i])

    data_info = joblib.load(os.path.join(data_path, 'data_info.pkl'))
    datasets = ['otto', 'dionis', 'helena', 'sf-crime', 'moa', 'scm20d', 'rf1', 'delicious', 'mediamill']

    tasks = product(datasets, ['tabnet'])

    # get runs
    if args.mode == 'run':
        Parallel(n_jobs=args.ngpus, backend="threading")(
            delayed(get_baseline_and_tune)(
                NAME, benchmark_path, data_path, ds, fr, rewrite=False) for (ds, fr) in tasks
        )

    # summarize
    tasks = [x for x in product(datasets, ['cb', 'xgb', 'tabnet'])]

    best_trials = {}

    for ds, fr in tasks:

        path = os.path.join(benchmark_path, NAME, ds, fr)

        best_val_score = np.inf
        best_test_score = np.inf
        best_trial = None
        num_trials = 0

        for trial in os.listdir(path):

            val_score, test_score, valid = get_result(os.path.join(path, trial))
            num_trials += valid

            if (val_score < best_val_score) and valid:
                best_val_score = val_score
                best_test_score = test_score
                best_trial = trial

        best_trials[(ds, fr)] = (best_val_score, best_test_score, best_trial, num_trials)

    summary = {}

    for ds, fr in best_trials:

        best_trial = best_trials[(ds, fr)][2]
        num_trials = best_trials[(ds, fr)][3]

        ds_sum = {'num_trials': num_trials}

        path = os.path.join(benchmark_path, NAME, ds, fr, best_trial)

        for i in range(5):
            folder = os.path.join(path, 'fold_{0}'.format(i))

            if i == 0:
                ds_sum['params'] = joblib.load(os.path.join(folder, 'params.pkl'))
                res = joblib.load(os.path.join(folder, 'results.pkl'))

                for k in res:
                    ds_sum[k] = [res[k]]

            else:
                res = joblib.load(os.path.join(folder, 'results.pkl'))
                for k in res:
                    ds_sum[k].append(res[k])

        summary[(ds, fr)] = ds_sum

        print(summary)

    joblib.dump(summary, os.path.join(benchmark_path, 'baselines_and_params.pkl'))
