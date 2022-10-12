import os

import joblib
from matplotlib import pyplot as plt
from pandas import DataFrame
from py_boost import SketchBoost
from sklearn.model_selection import KFold, train_test_split

data_path = 'data/processed'
benchmark_path = 'runs'

if __name__ == '__main__':

    params = joblib.load('runs/baselines_and_params.pkl')[('moa', 'cb')]['params']
    data_info = joblib.load(os.path.join(data_path, 'data_info.pkl'))
    # MOA dataset

    X_tot = joblib.load(os.path.join('data/processed', data_info['moa']['data']))
    y_tot = joblib.load(os.path.join('data/processed', data_info['moa']['target']))

    X, X_test, y, y_test = train_test_split(X_tot, y_tot, test_size=0.2, random_state=42)

    folds = KFold(5, shuffle=True, random_state=42)

    dd_0 = []

    for n, (f0, f1) in enumerate(folds.split(X)):

        model = SketchBoost('bce', 'bce', **params, sketch_method=None, es=0, ntrees=500, verbose=100)
        model.fit(X[f0], y[f0], eval_sets=[{'X': X[f1], 'y': y[f1]}])
        dd_0.append([x[0] for x in model.history])

        for i in [1, 5, 10, 20]:
            model = SketchBoost('bce', 'bce', **params, es=0,
                                sketch_method='rand', sketch_outputs=i, ntrees=500, verbose=100)
            model.fit(X[f0], y[f0], eval_sets=[{'X': X[f1], 'y': y[f1]}])
            dd_0.append([x[0] for x in model.history])
        break

    joblib.dump(dd_0, 'runs/moa_lk.pkl')

    # OTTO dataset

    X_tot = joblib.load(os.path.join('data/processed', data_info['otto']['data']))
    y_tot = joblib.load(os.path.join('data/processed', data_info['otto']['target']))

    X, X_test, y, y_test = train_test_split(X_tot, y_tot, test_size=0.2, random_state=42)

    folds = KFold(5, shuffle=True, random_state=42)

    dd_1 = []

    for n, (f0, f1) in enumerate(folds.split(X)):

        model = SketchBoost('crossentropy', 'crossentropy', **params,
                            sketch_method=None, es=0, ntrees=500, verbose=100)
        model.fit(X[f0], y[f0], eval_sets=[{'X': X[f1], 'y': y[f1]}])
        dd_1.append([x[0] for x in model.history])

        for i in [1, 2, 5, ]:
            model = SketchBoost('crossentropy', 'crossentropy', **params, es=0,
                                sketch_method='rand', sketch_outputs=i, ntrees=500, verbose=100)
            model.fit(X[f0], y[f0], eval_sets=[{'X': X[f1], 'y': y[f1]}])
            dd_1.append([x[0] for x in model.history])
        break

    joblib.dump(dd_1, 'runs/otto_lk.pkl')

    # VISUALIZE
    os.makedirs('output', exist_ok=True)

    for f, name, start, stop, grid in [
        ('runs/moa_lk.pkl', 'MoA', 10, 500, ['SketchBoost Full', 'k=1', 'k=5', 'k=10', 'k=20']),
        ('runs/otto_lk.pkl', 'Otto', 50, 350, ['SketchBoost Full', 'k=1', 'k=2', 'k=5'])
    ]:
        plt.figure()

        dat = DataFrame(joblib.load(f)).T[start:stop]
        dat.columns = grid
        dat.plot(figsize=(7, 5))

        plt.title('Learning Curve for {0} dataset'.format(name), fontsize=20)
        plt.legend(ncol=2, fontsize=15)
        plt.xlabel('Iteration', fontsize=15)
        plt.ylabel('Valid Error', fontsize=15)

        plt.grid()
        plt.savefig(os.path.join('output', name + '_lk.png'), dpi=720)
