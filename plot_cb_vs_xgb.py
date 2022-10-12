import argparse
import os

from time import time
from sklearn.datasets import make_classification

import catboost
import xgboost as xgb

from pandas import DataFrame

import joblib
from matplotlib import pyplot as plt

from py_boost import SketchBoost

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--skip', type=str, default='false')


def _train_cb(pool, n):
    res = {}

    model = catboost.CatBoostClassifier(
        objective='MultiClass',
        grow_policy='Depthwise',
        bootstrap_type='Bernoulli',
        subsample=1,
        border_count=256,
        iterations=n,
        max_depth=6,
        devices='0:0',
        learning_rate=0.01,
        l2_leaf_reg=1,
        min_data_in_leaf=1,
        # boost_from_average=True,
        score_function='L2',
        model_shrink_mode='Constant',
        task_type='GPU',
        allow_const_label=True,
        thread_count=8,
        verbose=100
    )

    t = time()

    model.fit(pool)

    res['train_time'] = time() - t

    return res


def train_cb(X_train, y_train):
    pool = catboost.Pool(X, label=y)

    res0 = _train_cb(pool, n=100)
    res1 = _train_cb(pool, n=200)

    return res1['train_time'] - res0['train_time']


def _train_xgb(params, dmat, n):
    res = {}

    # training
    t = time()
    xgb.train(params, dmat, num_boost_round=n)

    res['train_time'] = time() - t

    return res


def train_xgb(X_train, y_train):
    params = {

        'grow_policy': 'depthwise',
        'max_depth': 6,
        'eta': 0.01,
        'tree_method': 'gpu_hist',
        'subsample': 1,
        'colsample_bytree': 1,
        'colsample_bylevel': 1,
        'colsample_bynode': 1,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'num_class': y_train.max() + 1,
        'nthread': 8,
        'use_label_encoder': False

    }

    dmat = xgb.DMatrix(X, label=y)

    res0 = _train_xgb(params, dmat, n=100)
    res1 = _train_xgb(params, dmat, n=200)

    return res1['train_time'] - res0['train_time']


def _train_sketch(X_train, y_train, n):
    model = SketchBoost(
        'crossentropy',
        ntrees=n,
        lr=0.01,
        max_depth=6,
        colsample=1,
        subsample=1,
        lambda_l2=1,
        sketch_outputs=1,
        sketch_method='proj',
    )
    res = {}
    t = time()

    model.fit(X_train, y_train)

    res['train_time'] = time() - t

    return res


def train_sketch(X_train, y_train):
    res0 = _train_sketch(X_train, y_train, n=100)
    res1 = _train_sketch(X_train, y_train, n=200)

    return res1['train_time'] - res0['train_time']


if __name__ == '__main__':

    args = parser.parse_args()

    if args.skip == 'false':

        res = []

        for n in [5, 10, 25, 50, 100, 250, 500]:
            X, y = make_classification(n_samples=2000000, n_classes=n, n_features=100, n_informative=10, n_redundant=20)

            res_cb = train_cb(X, y)
            res_xgb = train_xgb(X, y)
            res_sketch = train_sketch(X, y)

            res.append((
                res_cb,
                res_xgb,
                res_sketch,
            ))

            print(
                n,
                res_cb,
                res_xgb,
                res_sketch
            )

        joblib.dump(res, 'runs/mo_example.pkl')

    res = joblib.load('runs/mo_example.pkl')

    # first fig

    df = DataFrame({

        'XGBoost': [x[1] for x in res],
        'CatBoost': [x[0] for x in res],

    }, index=[5, 10, 25, 50, 100, 250, 500])

    rects = df.plot(kind='bar', alpha=.6, edgecolor="black", hatch='//', figsize=(12, 7), width=0.85)

    for col, add in zip(['CatBoost', 'XGBoost'], [0.23, -0.21]):

        for index, row in enumerate(df[col].values):
            val = int(row)
            rects.text(index + add, row + 50, val, color='black', ha="center", fontsize=12)

    plt.legend(loc='best', fontsize=20)

    plt.ylim(top=df.values.max() * 1.1)
    plt.xticks(rotation=0, fontsize=25)
    plt.yticks(rotation=0, fontsize=25)
    plt.title('Training time for 100 iterations', fontsize=25)
    plt.xlabel('Number of Classes', fontsize=25)
    plt.ylabel('Time, sec', fontsize=25)
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/TrainTime.png', dpi=720)

    # last fig

    df = DataFrame({

        'XGBoost': [x[1] for x in res],
        'CatBoost': [x[0] for x in res],
        'SketchBoost': [x[2] for x in res],

    }, index=[5, 10, 25, 50, 100, 250, 500])

    rects = df.plot(kind='bar', alpha=.6, edgecolor="black", hatch='//', figsize=(12, 7), width=0.85)

    for col, add in zip(['CatBoost', 'XGBoost', 'SketchBoost'], [0.02, -0.31, 0.3]):

        for index, row in enumerate(df[col].values):
            val = int(row)
            rects.text(index + add, row + 50, val, color='black', ha="center", fontsize=12)

    plt.legend(loc='best', fontsize=20)

    plt.ylim(top=df.values.max() * 1.1)
    plt.xticks(rotation=0, fontsize=25)
    plt.yticks(rotation=0, fontsize=25)
    plt.title('Training time for 100 iterations', fontsize=25)
    plt.xlabel('Number of Classes', fontsize=25)
    plt.ylabel('Time, sec', fontsize=25)
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/TrainTimeSB.png', dpi=720)
