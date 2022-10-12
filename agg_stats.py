import joblib
import os
import numpy as np

from pandas import Series, DataFrame


def get_result(folder, attr, agg=np.mean):
    files = os.listdir(folder)

    stats = []

    for f in files:

        fname = os.path.join(folder, f, 'results.pkl')
        try:
            res = joblib.load(fname)
        except FileNotFoundError:
            continue

        stats.append(res[attr])

    valid = len(stats) == 5

    return agg(stats), valid


def get_pb_data_result(folder, data, strategy, attr, agg=np.mean):
    folder = os.path.join('runs', folder, data, 'pb')
    ls = os.listdir(folder)

    res = {}

    for f in filter(lambda x: strategy in x, ls):
        k = int(f.split('_')[-1])

        stat, valid = get_result(os.path.join(folder, f), attr, agg)

        assert valid

        res[k] = stat

    return Series(res).sort_index()


def get_baseline_result(folder, framework):
    res = {}
    for data in dataset_order:

        sub = {}

        for attr in ['train_time', 'best_iter', 'val_pred_time', 'test_pred_time',
                     'test_score', 'val_score', 'test_acc', 'val_acc']:

            if data in ['moa', 'delicious', 'mediamill', 'scm20d', 'rf1'] and framework == 'xgb':
                continue

            fold = os.path.join('runs', folder, data, framework, 'default')

            stat, valid = get_result(fold, attr)
            sub[attr] = stat
        res[data] = Series(sub)

    return DataFrame(res)


def min_max_scale(col):
    return (col - col.min()) / (col.max() - col.min())


def get_stat(fold):
    sub = {}

    for attr in stat_order:
        stat, valid = get_result(fold, attr)
        sub[attr] = stat

        stat, valid = get_result(fold, attr, lambda x: np.std(x, ddof=1))
        sub['std_' + attr] = stat

    return Series(sub)


if __name__ == '__main__':

    dataset_order = ['otto', 'dionis', 'helena', 'sf-crime', 'moa', 'moa', 'delicious', 'mediamill', 'scm20d', 'rf1']

    stat_order = ['train_time', 'best_iter', 'val_pred_time', 'test_pred_time',
                  'test_score', 'val_score', 'test_acc', 'val_acc']

    for data in dataset_order:

        # baselines
        os.makedirs(os.path.join('agg', data), exist_ok=True)
        res = {}

        for framework in ['xgb', 'cb']:

            fold = os.path.join('runs', 'BASELINES_REFIT', data, framework, 'default')
            try:
                sub = get_stat(fold)
            except FileNotFoundError:
                continue
            res[framework] = sub

        fold = os.path.join('runs', 'SKETCHBOOST_ALLK', data, 'pb', 'raw_10000')
        sub = get_stat(fold)
        res['pb'] = sub

        res = DataFrame(res).T
        joblib.dump(res, os.path.join('agg', data, 'baselines.pkl'))

        # strategy no hess
        for strategy in ['best', 'random', 'proj']:

            res = {}

            for attr in stat_order:
                res[attr] = get_pb_data_result('SKETCHBOOST_ALLK', data, strategy, attr)
                res['std_' + attr] = get_pb_data_result('SKETCHBOOST_ALLK', data, strategy, attr,
                                                        lambda x: np.std(x, ddof=1))

            res = DataFrame(res)
            # order by rank
            res['sorter_0'] = res['test_score'].rank() + res['train_time'].rank()
            res['sorter_1'] = min_max_scale(res['test_score']) + min_max_scale(res['train_time'])

            joblib.dump(res, os.path.join('agg', data, strategy + '.pkl'))

    for ds in ('caltech', 'nuswide', 'mnist', 'mnistreg'):

        gpath = os.path.join('runs', 'GBDTMO_BESTK', ds, 'gbdtmo')

        for task in ['raw', 'sparse']:

            task = [x for x in os.listdir(gpath) if x.startswith(task)][0]

            path = os.path.join('runs', 'GBDTMO_BESTK', ds, 'gbdtmo', task)
            for f in range(5):
                fpath = os.path.join(path, 'fold_' + str(f))

                res = joblib.load(os.path.join(fpath, 'results.pkl'))

                with open(os.path.join(fpath, 'train_log.txt')) as f:
                    log = f.readlines()
                    it = int(log[-1].split()[-1])

                res['best_iter'] = it
                joblib.dump(res, os.path.join(fpath, 'results.pkl'))

    dataset_order = ['caltech', 'nuswide', 'mnist', 'mnistreg']

    stat_order = ['train_time', 'best_iter', 'val_pred_time', 'test_pred_time',
                  'test_score', 'val_score', 'test_acc', 'val_acc']

    for data in dataset_order:

        # baselines
        os.makedirs(os.path.join('agg', data), exist_ok=True)
        res = {}
        path = os.path.join('runs', 'GBDTMO_BESTK', data, 'gbdtmo', )

        for st, g in zip(['raw', 'sparse'], ['gbdtmo', 'gbdtso']):
            st = [x for x in os.listdir(path) if x.startswith(st)][0]

            fold = os.path.join(path, st)

            sub = get_stat(fold)
            res[g] = sub

        fold = os.path.join('runs', 'GBDTMO_BESTK', data, 'pb', 'default')
        sub = get_stat(fold)
        res['pb'] = sub

        fold = os.path.join('runs', 'GBDTMO_BESTK', data, 'cb', 'default')
        sub = get_stat(fold)
        res['cb'] = sub

        res = DataFrame(res).T
        res['final_test_score'] = res['test_acc'] if data in ['yeast', 'mnist', 'caltech', 'nuswide'] else res[
            'test_score']
        res['std_final_test_score'] = res['std_test_acc'] if data in ['yeast', 'mnist', 'caltech', 'nuswide'] else res[
            'std_test_score']
        res['multiplier'] = -1 if data in ['yeast', 'mnist', 'caltech'] else 1
        joblib.dump(res, os.path.join('agg', data, 'baselines.pkl'))

        for strategy in ['random', 'proj']:

            res = {}

            for attr in stat_order:
                res[attr] = get_pb_data_result('GBDTMO_BESTK', data, strategy, attr)
                res['std_' + attr] = get_pb_data_result('GBDTMO_BESTK', data, strategy, attr,
                                                        lambda x: np.std(x, ddof=1))

            res = DataFrame(res)
            res['final_test_score'] = res['test_acc'] if data in ['yeast', 'mnist', 'caltech', 'nuswide'] else res[
                'test_score']
            res['std_final_test_score'] = res['std_test_acc'] if data in ['yeast', 'mnist', 'caltech', 'nuswide'] else \
                res['std_test_score']
            res['multiplier'] = -1 if data in ['yeast', 'mnist', 'caltech', 'nuswide'] else 1

            # order by rank
            res['sorter_def'] = res['final_test_score'] * res['multiplier']
            res['sorter_0'] = res['final_test_score'] * res['multiplier'].rank() + res['train_time'].rank()
            res['sorter_1'] = min_max_scale(res['final_test_score'] * res['multiplier']) + min_max_scale(
                res['train_time'])

            joblib.dump(res, os.path.join('agg', data, strategy + '.pkl'))
