import argparse
from copy import deepcopy

TIMEOUT = 3600 * 32

parser = argparse.ArgumentParser()

parser.add_argument('-b', '--bench', type=str)
parser.add_argument('-p', '--path', type=str)
parser.add_argument('-n', '--name', type=str)
parser.add_argument('-k', '--key', type=str)
parser.add_argument('-d', '--device', type=str)
parser.add_argument('-r', '--runner', type=str)


def objective(trial, params=None):
    trial_name = 'trial_{0}'.format(trial.number)
    if params is None:
        params = {}
    else:
        params = deepcopy(params)
    if args.runner != 'tabnet':
        if args.runner == 'xgb':
            params['min_data_in_leaf'] = trial.suggest_float("min_data_in_leaf", 1e-5, 10, log=True)

        else:
            params['min_data_in_leaf'] = trial.suggest_int("min_data_in_leaf", 1, 100, log=True)

        params['subsample'] = trial.suggest_float("subsample", 0.5, 1.0)
        params['max_depth'] = trial.suggest_int("max_depth", 3, 12)
        params['lambda_l2'] = trial.suggest_float("lambda_l2", .1, 50, log=True)
    else:
        params['lr'] = trial.suggest_float("lr", 1e-5, 0.1, log=True)

    res = run_cv_loop(args.name, args.device,
                      args.bench, args.path, args.key, args.runner, trial_name, params, rewrite=True)

    return np.mean([x['val_score'] for x in res])


if __name__ == '__main__':
    args = parser.parse_args()

    import optuna
    import numpy as np

    from utils import run_cv_loop

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30 if args.runner != 'tabnet' else 10, timeout=TIMEOUT)
