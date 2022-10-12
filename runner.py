import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-b', '--bench', type=str)
parser.add_argument('-p', '--path', type=str)
parser.add_argument('-k', '--key', type=str)
parser.add_argument('-f', '--fold', type=int)
parser.add_argument('-n', '--njobs', type=int)
parser.add_argument('-s', '--seed', type=int)
parser.add_argument('-d', '--device', type=str)
parser.add_argument('-o', '--output', type=str)
parser.add_argument('-r', '--runner', type=str)

LR = 0.05
NTREES = 5000
ES = 200


def softmax(x, clip_val=1e-7):
    exp_p = np.exp(x - x.max(axis=1, keepdims=True))

    return np.clip(exp_p / exp_p.sum(axis=1, keepdims=True), clip_val, 1 - clip_val)


def sigmoid(x, clip_val=1e-7):
    return np.clip(1 / (1 + np.exp(- x)), clip_val, 1 - clip_val)


def cent(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

    return -np.log(np.take_along_axis(y_pred, y_true[:, np.newaxis].astype(np.int32), axis=1)).mean()


def bce(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

    return -np.log(y_true * y_pred + (1 - y_true) * (1 - y_pred)).mean()


def rmse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean() ** .5


def r2_score(y_true, y_pred):
    var = np.power((y_true - y_true.mean()), 2).mean(axis=0)
    mod = np.power((y_true - y_pred), 2).mean(axis=0)

    return (1 - mod / var).mean()


def evaluate(y_test, test_pred, y_val, val_pred, task, results):
    if task == 'multiclass':
        results['test_score'] = cent(y_test, test_pred)
        results['val_score'] = cent(y_val, val_pred)

        results['test_acc'] = (y_test.astype(np.int32) == test_pred.argmax(axis=1)).mean()
        results['val_acc'] = (y_val.astype(np.int32) == val_pred.argmax(axis=1)).mean()

    elif task == 'multilabel':

        results['test_score'] = bce(y_test, test_pred)
        results['val_score'] = bce(y_val, val_pred)

        results['test_acc'] = (y_test.astype(np.int32) == (test_pred > .5).astype(np.int32)).mean()
        results['val_acc'] = (y_val.astype(np.int32) == (val_pred > .5).astype(np.int32)).mean()

    else:

        results['test_score'] = rmse(y_test, test_pred)
        results['val_score'] = rmse(y_val, val_pred)

        results['test_acc'] = r2_score(y_test, test_pred)
        results['val_acc'] = r2_score(y_val, val_pred)

    return results


def train_pb(X_train, y_train, X_val, y_val, X_test, y_test, task, params):
    # get task
    if task == 'multiclass':
        loss = 'crossentropy'
        metric = 'crossentropy' if 'acc' not in params else 'accuracy'

    elif task == 'multilabel':
        loss = 'bce'
        metric = 'bce'

    else:
        loss = 'mse'
        metric = 'rmse'

    # get sketch method

    method = params['method']
    k = params['dim']
    use_hess = False if 'use_hess' not in params else params['use_hess']

    if method == 'raw':
        proxy = None

    elif method == 'best':
        proxy = 'topk'

    elif method == 'random':
        proxy = 'rand'

    elif method == 'proj':
        proxy = 'proj'

    else:
        raise ValueError('Unknown proxy method')

    # init 

    results = {}

    model = SketchBoost(loss,
                        metric,
                        ntrees=params['ntrees'],
                        lr=params['lr'],
                        verbose=100,
                        max_bin=256 if 'max_bin' not in params else params['max_bin'],
                        es=params['es'],
                        lambda_l2=params['lambda_l2'],
                        colsample=1,
                        subsample=params['subsample'],
                        max_depth=params['max_depth'],
                        min_data_in_leaf=params['min_data_in_leaf'],
                        sketch_outputs=k,
                        sketch_method=proxy,
                        use_hess=use_hess
                        )
    # training
    t = time()
    model.fit(X_train, y_train, eval_sets=[{'X': X_val, 'y': y_val}])
    results['train_time'] = time() - t
    results['best_iter'] = model.best_round

    # val prediction

    t = time()
    val_pred = model.predict(X_val)
    results['val_pred_time'] = time() - t

    # test prediction

    t = time()
    test_pred = model.predict(X_test)
    results['test_pred_time'] = time() - t

    # evaluate

    results = evaluate(y_test, test_pred, y_val, val_pred, task, results)

    return results


def train_gbdtmo(X_train, y_train, X_val, y_val, X_test, y_test, task, nout, params):
    LIB = gbdtmo.load_lib('GBDTMO/build/gbdtmo.so')

    # get task
    if task == 'multiclass':
        loss = b'ce'
    else:
        loss = b'mse'

    # get sketch method

    p = {'max_depth': params['max_depth'],
         'max_leaves': int(0.75 * 2 ** params['max_depth']),
         'topk': params['dim'],
         'loss': loss,
         'gamma': params['min_gain_to_split'],
         'num_threads': args.njobs,
         'max_bins': params['max_bin'],
         'lr': params['lr'],
         'reg_l2': params['lambda_l2'],
         'early_stop': params['es'],
         'one_side': params['one_side'],
         'verbose': True,
         'min_samples': params['min_data_in_leaf']}

    # init 

    results = {}

    model = gbdtmo.GBDTMulti(LIB, out_dim=nout, params=p)
    model.set_data((X_train, y_train), (X_val, y_val))

    # training
    t = time()
    model.train(params['ntrees'])
    results['train_time'] = time() - t
    results['best_iter'] = None

    # val prediction

    t = time()
    val_pred = model.predict(X_val)
    results['val_pred_time'] = time() - t

    # test prediction

    t = time()
    test_pred = model.predict(X_test)
    results['test_pred_time'] = time() - t

    if task == 'multiclass':
        val_pred = softmax(val_pred)
        test_pred = softmax(test_pred)

    elif task == 'multilabel':
        val_pred = np.clip(val_pred, 1e-7, 1 - 1e-7)
        test_pred = np.clip(test_pred, 1e-7, 1 - 1e-7)

    # evaluate
    results = evaluate(y_test, test_pred, y_val, val_pred, task, results)

    return results


def train_cb(X_train, y_train, X_val, y_val, X_test, y_test, task, params):
    # get task
    learner = catboost.CatBoostClassifier
    if task == 'multiclass':
        loss = 'MultiClass'
        metric = 'MultiClass' if 'acc' not in params else 'Accuracy'
        task_type = 'GPU' if 'cpu' not in params else 'CPU'


    elif task == 'multilabel':
        loss = 'MultiLogloss'
        metric = 'MultiLogloss'
        task_type = 'CPU'

    else:
        learner = catboost.CatBoostRegressor
        loss = 'MultiRMSE'
        metric = 'MultiRMSE'
        task_type = 'CPU'

    # init 

    results = {}

    model = learner(
        objective=loss,
        eval_metric=metric,
        grow_policy='Depthwise',
        bootstrap_type='Bernoulli',
        subsample=params['subsample'],
        border_count=256 if 'max_bin' not in params else params['max_bin'],
        iterations=params['ntrees'],
        od_wait=params['es'],
        max_depth=params['max_depth'],
        devices='0:0',
        learning_rate=params['lr'],
        l2_leaf_reg=params['lambda_l2'],
        min_data_in_leaf=params['min_data_in_leaf'],
        # boost_from_average=True,
        score_function='L2',
        model_shrink_mode='Constant',
        task_type=task_type,
        allow_const_label=True,
        thread_count=args.njobs,
        verbose=100
    )
    # training
    t = time()

    model.fit(X_train, y_train,
              eval_set=(X_val, y_val)
              )

    results['train_time'] = time() - t
    results['best_iter'] = model.best_iteration_ + 1

    # val prediction

    t = time()
    if task == 'multitask':
        val_pred = model.predict(X_val)
    else:
        val_pred = model.predict_proba(X_val)
    results['val_pred_time'] = time() - t

    # test prediction

    t = time()
    if task == 'multitask':
        test_pred = model.predict(X_test)
    else:
        test_pred = model.predict_proba(X_test)

    results['test_pred_time'] = time() - t

    # evaluate

    results = evaluate(y_test, test_pred, y_val, val_pred, task, results)

    return results


def train_xgb(X_train, y_train, X_val, y_val, X_test, y_test, task, nout, params):
    # get task
    if task == 'multiclass':
        estimator = xgb.XGBClassifier
        num_class = nout
        eval_metric = 'mlogloss'
    elif task == 'multilabel':
        estimator = xgb.XGBClassifier
        num_class = None
        eval_metric = 'logloss'
    else:
        estimator = xgb.XGBRegressor
        num_class = None
        eval_metric = 'rmse'

    # init 

    results = {}

    model = estimator(

        n_estimators=params['ntrees'],
        grow_policy='depthwise',
        max_depth=params['max_depth'],
        learning_rate=params['lr'],
        min_child_weight=params['min_data_in_leaf'],
        tree_method='gpu_hist',
        subsample=params['subsample'],
        colsample_bytree=1.,
        colsample_bylevel=1.,
        colsample_bynode=1.,
        reg_alpha=0,
        reg_lambda=params['lambda_l2'],
        num_class=num_class,
        nthread=args.njobs,
        use_label_encoder=False
    )

    # training
    t = time()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              eval_metric=eval_metric, early_stopping_rounds=params['es'], verbose=100)

    results['train_time'] = time() - t
    results['best_iter'] = model.best_iteration + 1

    # val prediction

    t = time()

    if task in ['multiclass', 'multilabel']:
        val_pred = model.predict_proba(X_val)
    else:
        val_pred = model.predict(X_val)

    results['val_pred_time'] = time() - t

    # test prediction

    t = time()

    if task in ['multiclass', 'multilabel']:
        test_pred = model.predict_proba(X_test)
    else:
        test_pred = model.predict(X_test)

    results['test_pred_time'] = time() - t

    # evaluate
    results = evaluate(y_test, test_pred, y_val, val_pred, task, results)

    return results


def train_tabnet(X_train, y_train, X_val, y_val, X_test, y_test, task, params):
    MAX_EPOCHS = 500
    EARLY_STOPING = 16
    print(params)
    bs = 256
    nrow = X_train.shape[0]
    if nrow > 50000:
        bs = 512
    if nrow > 100000:
        bs = 1024
    bs_pretrain = bs * 4
    default_pretrain_params = {
        "binary": {
            "fit_params": {
                "batch_size": bs_pretrain,
                "virtual_batch_size": np.minimum(bs_pretrain, 512),
                'max_epochs': MAX_EPOCHS,
                'patience': EARLY_STOPING,
                'pretraining_ratio': 0.8
            },
            "init_params": {
                'momentum': 0.7,
                'n_d': 64, 'n_a': 64, 'n_steps': 10,
                'gamma': 1.5, 'n_independent': 2, 'n_shared': 2,
                'cat_emb_dim': 1,
                'lambda_sparse': 1e-4, 'clip_value': 2.,
                'optimizer_fn': torch.optim.Adam,
                'optimizer_params': {'lr': 2e-2},
                'scheduler_params': {"gamma": 0.95,
                                     "step_size": 25},
                'scheduler_fn': torch.optim.lr_scheduler.StepLR,
                'epsilon': 1e-15,
                'mask_type': "entmax"
            }
        },
        "reg": {
            "fit_params": {
                "batch_size": bs_pretrain,
                "virtual_batch_size": np.minimum(bs_pretrain, 128),
                'max_epochs': MAX_EPOCHS,
                'patience': EARLY_STOPING,
                'pretraining_ratio': 0.8
            },
            "init_params": {
                'momentum': 0.7,
                'n_d': 64, 'n_a': 64, 'n_steps': 7,
                'gamma': 1.5, 'n_independent': 2, 'n_shared': 2,
                'cat_emb_dim': 1,
                'lambda_sparse': 1e-4, 'clip_value': 2.,
                'optimizer_fn': torch.optim.Adam,
                'optimizer_params': {'lr': 1e-2},
                'scheduler_params': {"gamma": 0.95,
                                     "step_size": 10},
                'scheduler_fn': torch.optim.lr_scheduler.StepLR,
                'epsilon': 1e-15,
                'mask_type': "entmax"
            }
        },
    }

    default_model_params = {
        "binary": {
            "fit_params": {
                "batch_size": bs,
                "virtual_batch_size": np.minimum(bs, 512),
                'max_epochs': MAX_EPOCHS,
                'patience': EARLY_STOPING,
                # "augmentations": None,
                # "aug_p": 0.2,
            },
            "init_params": {
                'momentum': 0.7,
                'n_d': 64, 'n_a': 64, 'n_steps': 5,
                'gamma': 1.5, 'n_independent': 2, 'n_shared': 2,
                'cat_emb_dim': 1,
                'lambda_sparse': 1e-4, 'clip_value': 2.,
                'optimizer_fn': torch.optim.Adam,
                'optimizer_params': {'lr': params.get('lr', 2e-2)},
                'scheduler_params': {"gamma": 0.95,
                                     "step_size": 25},
                'scheduler_fn': torch.optim.lr_scheduler.StepLR,
                'epsilon': 1e-15,
            }
        },
        "reg": {
            "fit_params": {
                "batch_size": bs,
                "virtual_batch_size": np.minimum(bs, 128),
                'max_epochs': MAX_EPOCHS,
                'patience': EARLY_STOPING,
                # "augmentations": None,
                # "aug_p": 0.2,
            },
            "init_params": {
                'momentum': 0.7,
                'n_d': 64, 'n_a': 64, 'n_steps': 7,
                'gamma': 1.5, 'n_independent': 2, 'n_shared': 2,
                'cat_emb_dim': 1,
                'lambda_sparse': 1e-4, 'clip_value': 2.,
                'optimizer_fn': torch.optim.Adam,
                'optimizer_params': {'lr': params.get('lr', 1e-2)},
                'scheduler_params': {"gamma": 0.95,
                                     "step_size": 10},
                'scheduler_fn': torch.optim.lr_scheduler.StepLR,
                'epsilon': 1e-15,
            }
        },
    }

    model_to_task = {
        "binary": TabNetClassifier,
        "reg": TabNetRegressor,
        "multiclass": TabNetClassifier,
        "multilabel": TabNetMultiTaskClassifier,
        "multi:reg": TabNetRegressor,
    }

    default_pretrain_params["multiclass"] = default_pretrain_params["binary"]
    default_pretrain_params["multilabel"] = default_pretrain_params["binary"]
    default_pretrain_params["multi:reg"] = default_pretrain_params["reg"]

    default_model_params["multiclass"] = default_model_params["binary"]
    default_model_params["multilabel"] = default_model_params["binary"]
    default_model_params["multi:reg"] = default_model_params["reg"]

    # fill na with mean
    X_val = np.where(np.isnan(X_val), np.ma.array(X_train, mask=np.isnan(X_train)).mean(axis=0), X_val)
    X_test = np.where(np.isnan(X_test), np.ma.array(X_train, mask=np.isnan(X_train)).mean(axis=0), X_test)
    X_train = np.where(np.isnan(X_train), np.ma.array(X_train, mask=np.isnan(X_train)).mean(axis=0), X_train)
    if task == 'multiclass':
        LABELS = sorted(list(np.unique(y_train)))
    else:
        LABELS = [0, 1]

    class LogLoss(Metric):
        def __init__(self):
            self._name = "log_loss"
            self._maximize = False

        def __call__(self, y_true, y_score):
            ll = log_loss(y_true, 1 - y_score[:, 0], labels=LABELS, eps=1e-5)
            return ll

    class LogLossMultiClass(Metric):
        def __init__(self):
            self._name = "log_loss_multi"
            self._maximize = False

        def __call__(self, y_true, y_score):
            ll = log_loss(y_true, y_score, labels=LABELS)
            return ll

    task_to_metric = {
        "binary": "auc",
        "reg": "mse",
        "multiclass": LogLossMultiClass,
        "multilabel": LogLoss,
        "multi:reg": "mse",
    }

    # get task
    if (task != 'multiclass') and (task != 'multilabel'):
        task = "multi:reg"

    # init
    cat_idxs = []
    cat_dims = []
    num_idxs = np.arange(X_train.shape[1])

    results = {}
    # unsupervised pretraining
    p = default_pretrain_params[task]
    pretrain = TabNetPretrainer(**p["init_params"])
    t_pr = time()
    pretrain.fit(X_train=X_train.astype(np.float32),
                 eval_set=[X_val.astype(np.float32)],
                 eval_name=['valid'],
                 **p["fit_params"])
    t_pr = time() - t_pr
    p = default_model_params[task]
    model = model_to_task[task](**p["init_params"])
    # training
    t = time()
    y_train_new = y_train.astype(int).copy()
    y_val_new = y_val.astype(int).copy()
    print(task)
    if task == 'multilabel':
        # add fake ones or zeros in binary case
        for i in range(y_train.shape[1]):
            tr_s = set(y_train[:, i])
            vl_s = set(y_val[:, i])
            if len(tr_s) == len(vl_s):
                if len(vl_s) == 1:
                    if 1 in vl_s:
                        y_train_new[0, i] = 0
                        y_val_new[0, i] = 0
                    else:
                        y_train_new[0, i] = 1
                        y_val_new[0, i] = 1
            elif len(tr_s) == 2 and len(vl_s) == 1:
                if 1 in vl_s:
                    y_val_new[0, i] = 0
                else:
                    y_val_new[0, i] = 1
            elif len(tr_s) == 1 and len(vl_s) == 2:
                if 1 in tr_s:
                    y_train_new[0, i] = 0
                else:
                    y_train_new[0, i] = 1

    model.fit(X_train=X_train.astype(np.float32), y_train=y_train_new,
              eval_set=[(X_val.astype(np.float32), y_val_new)],
              eval_name=['valid'],
              eval_metric=[task_to_metric[task]],
              from_unsupervised=pretrain,
              **p["fit_params"])

    results['train_time'] = time() - t + t_pr
    results['best_iter'] = model.best_epoch

    # val prediction

    t = time()
    if task == "multiclass":
        val_pred = model.predict_proba(X_val)
    elif task == 'multilabel':
        val_pred = model.predict_proba(X_val)
        if isinstance(val_pred, list):
            val_pred = np.vstack([i[:, 1] if i.shape[1] == 2 else 1 - i[:, 0] for i in val_pred])
            val_pred = np.moveaxis(val_pred, 0, 1)
    elif task == "binary":
        val_pred = model.predict_proba(X_val)[:, 1]
    else:
        val_pred = model.predict(X_val)

    results['val_pred_time'] = time() - t

    # test prediction

    t = time()
    if task == "multiclass":
        test_pred = model.predict_proba(X_test)
    elif task == 'multilabel':
        test_pred = model.predict_proba(X_test)
        if isinstance(test_pred, list):
            test_pred = np.stack([i[:, 1] if i.shape[1] == 2 else 1 - i[:, 0] for i in test_pred])
            test_pred = np.moveaxis(test_pred, 0, 1)
    elif task == "binary":
        test_pred = model.predict_proba(X_test)[:, 1]
    else:
        test_pred = model.predict(X_test)

    results['test_pred_time'] = time() - t

    # evaluate
    results = evaluate(y_test, test_pred, y_val, val_pred, task, results)
    return results


if __name__ == '__main__':

    import os

    args = parser.parse_args()
    str_nthr = str(args.njobs)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    os.environ["OMP_NUM_THREADS"] = str_nthr  # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = str_nthr  # export OPENBLAS_NUM_THREADS=4
    os.environ["MKL_NUM_THREADS"] = str_nthr  # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = str_nthr  # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = str_nthr  # export NUMEXPR_NUM_THREADS=6

    import numpy as np
    import joblib

    np.random.seed(args.seed)

    import catboost
    import xgboost as xgb
    import gbdtmo

    from sklearn.model_selection import train_test_split, KFold
    from py_boost import SketchBoost

    import torch
    from pytorch_tabnet.pretraining import TabNetPretrainer
    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
    from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
    from pytorch_tabnet.metrics import Metric
    from sklearn.metrics import log_loss
    from time import time

    # paths .. 
    data_info = joblib.load(os.path.join(args.path, 'data_info.pkl'))[args.key]
    _params = joblib.load(os.path.join(args.output, 'params.pkl'))

    if args.runner != 'tabnet':
        params = {**{'lr': LR, 'ntrees': NTREES, 'es': ES}, **_params}
    else:
        params = {**_params}

    print('Params to train')
    print(params)

    X_tot = joblib.load(os.path.join(args.path, data_info['data']))
    y_tot = joblib.load(os.path.join(args.path, data_info['target']))

    if 'split' in data_info:
        tr, ts = joblib.load(os.path.join(args.path, data_info['split']))
        X, X_test, y, y_test = X_tot[tr], X_tot[ts], y_tot[tr], y_tot[ts]
    else:
        X, X_test, y, y_test = train_test_split(X_tot, y_tot, test_size=0.2, random_state=args.seed)

    folds = KFold(5, shuffle=True, random_state=args.seed)

    for n, (f0, f1) in enumerate(folds.split(X)):

        if n != args.fold:
            continue

        if args.runner == 'pb':
            results = train_pb(X[f0], y[f0], X[f1], y[f1], X_test, y_test, data_info['task_type'], params)

        if args.runner == 'cb':
            results = train_cb(X[f0], y[f0], X[f1], y[f1], X_test, y_test, data_info['task_type'], params)

        if args.runner == 'xgb':
            results = train_xgb(X[f0], y[f0], X[f1], y[f1], X_test, y_test,
                                data_info['task_type'], data_info['nout'], params)

        if args.runner == 'gbdtmo':
            results = train_gbdtmo(X[f0], y[f0], X[f1], y[f1], X_test, y_test, data_info['task_type'],
                                   data_info['nout'], params)

        if args.runner == 'tabnet':
            results = train_tabnet(X[f0], y[f0], X[f1], y[f1], X_test, y_test, data_info['task_type'], params)

    joblib.dump(results, os.path.join(args.output, 'results.pkl'))
