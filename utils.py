import os
import subprocess
import sys

import joblib

RUNNER_PATH = 'runner.py'
DEBUG = True

HARD_TIMEOUT = 3600 * 300
NTHREADS = 8
SEED = 42


def run_task(name, gpu, benchmark_path, data_path, dataset, task, fold, trial_name, params, rewrite=False):
    run_name = dataset + '_' + task + '_' + trial_name + '_' + str(fold)

    print('Processing', run_name)

    benchmark_path = os.path.abspath(benchmark_path)
    data_path = os.path.abspath(data_path)
    output = os.path.join(benchmark_path, name, dataset, task, str(trial_name), 'fold_{0}'.format(fold))

    os.makedirs(output, exist_ok=True)

    success_flg = os.path.join(output, 'SUCCESS')

    if os.path.exists(success_flg) and not rewrite:
        return

        # clean folder
    for f in (x for x in os.listdir(output) if not x.startswith('.')):
        os.remove(os.path.join(output, f))

    joblib.dump(params, os.path.join(output, 'params.pkl'))

    # TRAIN
    try:

        script = ""

        log = subprocess.check_output(script + ' '.join([
            sys.executable,
            RUNNER_PATH,
            '-b', benchmark_path,
            '-p', data_path,
            '-k', dataset,
            '-f', str(fold),
            '-n', str(NTHREADS),
            '-s', str(SEED),
            '-d', ','.join(map(str, gpu)),
            '-o', output,
            '-r', task

        ]), shell=True, stderr=subprocess.STDOUT, executable='/bin/bash').decode()

        if DEBUG:
            print(log)

        with open(success_flg, 'w') as f:
            pass

        with open(os.path.join(output, 'train_log.txt'), 'w') as f:
            f.write(log)

    except subprocess.CalledProcessError as e:

        print(e.output.decode())

        with open(os.path.join(output, 'ERROR'), 'w') as f:
            pass

        with open(os.path.join(output, 'train_log.txt'), 'w') as f:
            f.write(e.output.decode())

    except subprocess.TimeoutExpired:

        with open(os.path.join(output, 'TIMEOUT'), 'w') as f:
            pass

        print('HARD TIMEOUT!')

    results = joblib.load(os.path.join(output, 'results.pkl'))
    return results


def run_cv_loop(name, gpu, benchmark_path, data_path, dataset, task, trial_name, params, rewrite=False):
    res = []

    for i in range(5):
        results = run_task(name, gpu, benchmark_path, data_path, dataset, task, i, trial_name, params, rewrite=rewrite)

        res.append(results)

    return res
