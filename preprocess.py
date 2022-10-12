import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.append('GBDTMO-EX')
import loader

from pandas import Series

original_data = 'data'
processed_data = 'data/processed'


def get_split(path, n=0):
    idx = []

    with open(os.path.join(original_data, path)) as f:
        split = f.readlines()
        for sp in split:
            idx.append(int(sp.split()[n]))

    return np.array(idx) - 1


def get_data(path):
    with open(os.path.join(original_data, path)) as f:
        raw_data = f.readlines()

    rows, feats, labels = map(int, raw_data[0].split())

    X, y = np.zeros((rows, feats), dtype=np.float32), np.zeros((rows, labels), dtype=np.float32)

    for n, row in enumerate(raw_data[1:]):
        row = row.split()

        if ':' not in row[0]:
            y[n, list(map(int, row[0].split(',')))] = 1
            row = row[1:]

        row = list(map(lambda x: x.split(':'), row))
        X[n, [int(x[0]) for x in row]] = [float(x[1]) for x in row]

    return X, y


if __name__ == '__main__':

    ### MOA

    data_path = os.path.join(processed_data, 'moa')
    os.makedirs(data_path, exist_ok=True)


    def preprocess_data(data, *targets):

        data = data.copy()
        data['cp_type'] = data['cp_type'] == 'ctl_vehicle'
        data['cp_dose'] = data['cp_dose'] == 'D1'
        X = data.drop('sig_id', axis=1).values.astype(np.float32)

        if len(targets) == 0:
            return X

        y = np.concatenate(
            [x.set_index('sig_id').loc[data['sig_id'].values].values.astype(np.float32) for x in targets], axis=1)

        return X, y


    data = pd.read_csv(os.path.join(original_data, 'lish_moa/train_features.csv'))
    scored = pd.read_csv(os.path.join(original_data, 'lish_moa/train_targets_scored.csv'))

    X, y = preprocess_data(data, scored)

    joblib.dump(X, os.path.join(data_path, 'feats.pkl'))
    joblib.dump(y, os.path.join(data_path, 'target.pkl'))

    print(y.shape)

    ### DIONIS

    data_path = os.path.join(processed_data, 'dionis')
    os.makedirs(data_path, exist_ok=True)

    data = pd.read_csv(os.path.join(original_data, 'dionis.csv'))
    X, y = data.drop('class', axis=1).values.astype(np.float32), data['class'].values.astype(np.float32)

    joblib.dump(X, os.path.join(data_path, 'feats.pkl'))
    joblib.dump(y, os.path.join(data_path, 'target.pkl'))

    print(y.max() + 1)

    ### SF-CRIME

    data_path = os.path.join(processed_data, 'sf-crime')
    os.makedirs(data_path, exist_ok=True)


    def label_encode(col):

        un = col.value_counts().index.values
        return col.map(Series(np.arange(un.shape[0]), index=un)).values.astype(np.int32)


    def preprocess_data(data):

        y = label_encode(data['Category'])

        data = data[['Dates', 'PdDistrict', 'Address', 'X', 'Y']].copy()

        for col in ['PdDistrict', 'Address']:
            data[col] = label_encode(data[col])

        data['Dates'] = pd.to_datetime(data['Dates'])

        data['year'] = data['Dates'].dt.year
        data['month'] = data['Dates'].dt.month
        data['day'] = data['Dates'].dt.day
        data['wd'] = data['Dates'].dt.weekday
        data['hour'] = data['Dates'].dt.hour
        data['minute'] = data['Dates'].dt.minute

        X = data.drop('Dates', axis=1).values.astype(np.float32)

        return X, y


    data = pd.read_csv(os.path.join(original_data, 'sf-crime/train.csv.zip'))
    X, y = preprocess_data(data)

    joblib.dump(X, os.path.join(data_path, 'feats.pkl'))
    joblib.dump(y, os.path.join(data_path, 'target.pkl'))

    print(y.max() + 1)

    ### HELENA

    data_path = os.path.join(processed_data, 'helena')
    os.makedirs(data_path, exist_ok=True)

    data = pd.read_csv(os.path.join(original_data, 'helena.csv'))
    X, y = data.drop('class', axis=1).values.astype(np.float32), data['class'].values.astype(np.float32)

    joblib.dump(X, os.path.join(data_path, 'feats.pkl'))
    joblib.dump(y, os.path.join(data_path, 'target.pkl'))

    print(y.max() + 1)

    ### OTTO

    data_path = os.path.join(processed_data, 'otto')
    os.makedirs(data_path, exist_ok=True)


    def preprocess_data(data):

        X = data.drop(['id', 'target'], axis=1).values.astype(np.float32)
        y = data['target'].map(lambda x: x[-1]).values.astype(np.int32) - 1

        return X, y


    data = pd.read_csv(os.path.join(original_data, 'otto/train.csv'))
    X, y = preprocess_data(data)

    joblib.dump(X, os.path.join(data_path, 'feats.pkl'))
    joblib.dump(y, os.path.join(data_path, 'target.pkl'))

    print(y.max() + 1)

    ### SCM20D

    data_path = os.path.join(processed_data, 'scm20d')
    os.makedirs(data_path, exist_ok=True)

    data = pd.read_csv(os.path.join(original_data, 'scm20d.csv'))

    ycols = ['LBL', 'MTLp2A', 'MTLp3A', 'MTLp4A', 'MTLp5A', 'MTLp6A', 'MTLp7A', 'MTLp8A',
             'MTLp9A', 'MTLp10A', 'MTLp11A', 'MTLp12A', 'MTLp13A', 'MTLp14A',
             'MTLp15A', 'MTLp16A']

    X, y = data.drop(ycols, axis=1).values.astype(np.float32), data[ycols].values.astype(np.float32)

    joblib.dump(X, os.path.join(data_path, 'feats.pkl'))
    joblib.dump(y, os.path.join(data_path, 'target.pkl'))

    print(y.shape)

    ### RF1 

    data_path = os.path.join(processed_data, 'rf1')
    os.makedirs(data_path, exist_ok=True)

    data = pd.read_csv(os.path.join(original_data, 'rf1.csv'), na_values='?')
    ycols = ['CHSI2_48H__0', 'NASI2_48H__0', 'EADM7_48H__0', 'SCLM7_48H__0', 'CLKM7_48H__0',
             'VALI2_48H__0', 'NAPM7_48H__0', 'DLDI4_48H__0']

    X, y = data.drop(ycols, axis=1).values.astype(np.float32), data[ycols].values.astype(np.float32)

    joblib.dump(X, os.path.join(data_path, 'feats.pkl'))
    joblib.dump(y, os.path.join(data_path, 'target.pkl'))

    print(y.shape)

    ### DELICIOUS

    data_path = os.path.join(processed_data, 'delicious')
    os.makedirs(data_path, exist_ok=True)

    X, y = get_data('Delicious/Delicious_data.txt')
    split = get_split('Delicious/delicious_trSplit.txt'), get_split('Delicious/delicious_tstSplit.txt')

    joblib.dump(X, os.path.join(data_path, 'feats.pkl'))
    joblib.dump(y, os.path.join(data_path, 'target.pkl'))
    joblib.dump(split, os.path.join(data_path, 'split.pkl'))

    print(y.shape)

    ### MEDIAMILL

    data_path = os.path.join(processed_data, 'mediamill')
    os.makedirs(data_path, exist_ok=True)

    X, y = get_data('Mediamill/Mediamill_data.txt')
    split = get_split('Mediamill/mediamill_trSplit.txt'), get_split('Mediamill/mediamill_tstSplit.txt')

    joblib.dump(X, os.path.join(data_path, 'feats.pkl'))
    joblib.dump(y, os.path.join(data_path, 'target.pkl'))
    joblib.dump(split, os.path.join(data_path, 'split.pkl'))

    print(y.shape)

    ### SUMMARY

    data_info = {

        'moa': {
            'data': 'moa/feats.pkl',
            'target': 'moa/target.pkl',
            'nout': 206,
            'task_type': 'multilabel',

        },

        'dionis': {
            'data': 'dionis/feats.pkl',
            'target': 'dionis/target.pkl',
            'nout': 355,
            'task_type': 'multiclass',

        },

        'sf-crime': {
            'data': 'sf-crime/feats.pkl',
            'target': 'sf-crime/target.pkl',
            'nout': 39,
            'task_type': 'multiclass',

        },

        'helena': {
            'data': 'helena/feats.pkl',
            'target': 'helena/target.pkl',
            'nout': 100,
            'task_type': 'multiclass',

        },

        'otto': {
            'data': 'otto/feats.pkl',
            'target': 'otto/target.pkl',
            'nout': 9,
            'task_type': 'multiclass',

        },

        'scm20d': {
            'data': 'scm20d/feats.pkl',
            'target': 'scm20d/target.pkl',
            'nout': 16,
            'task_type': 'multitask',

        },

        'rf1': {
            'data': 'rf1/feats.pkl',
            'target': 'rf1/target.pkl',
            'nout': 8,
            'task_type': 'multitask',

        },

        'delicious': {
            'data': 'delicious/feats.pkl',
            'target': 'delicious/target.pkl',
            'nout': 983,
            'task_type': 'multilabel',

            'split': 'delicious/split.pkl',

        },

        'mediamill': {
            'data': 'mediamill/feats.pkl',
            'target': 'mediamill/target.pkl',
            'nout': 101,
            'task_type': 'multilabel',

            'split': 'mediamill/split.pkl',

        },

    }

    ### GBDTMO datasets

    base_path = 'GBDTMO-EX/dataset'
    raw_names = ['Caltech101.npz', 'nus-wide.npz', 'mnist.npz', 'mnist.npz', ]
    process_fns = [loader.Caltech101, loader.nus, loader.mnist_cls, loader.mnist_reg]
    aliases = ['caltech', 'nuswide', 'mnist', 'mnistreg']
    tasks = ['multiclass', 'multilabel', 'multiclass', 'multitask']

    for raw, fn, alias, task in zip(raw_names, process_fns, aliases, tasks):
        data_path = os.path.join(processed_data, alias)
        os.makedirs(data_path, exist_ok=True)

        x_train, y_train, x_test, y_test = fn(os.path.join(base_path, raw))
        split = np.arange(x_train.shape[0]), np.arange(x_test.shape[1]) + x_train.shape[0]
        X, y = np.concatenate([x_train, x_test], axis=0), np.concatenate([y_train, y_test], axis=0)

        joblib.dump(X, os.path.join(data_path, 'feats.pkl'))
        joblib.dump(y, os.path.join(data_path, 'target.pkl'))
        joblib.dump(split, os.path.join(data_path, 'split.pkl'))

        data_info[alias] = {
            'data': os.path.join(alias, 'feats.pkl'),
            'target': os.path.join(alias, 'target.pkl'),
            'nout': y.max() + 1 if task == 'multiclass' else y.shape[1],
            'task_type': task,

            'split': os.path.join(alias, 'split.pkl'),

        }

        print(data_info[alias]['nout'])

    ### SAVE SUMMARY
    print(data_info)
    joblib.dump(data_info, os.path.join(processed_data, 'data_info.pkl'))
