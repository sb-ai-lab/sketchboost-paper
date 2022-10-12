# GBDTMO-EX
This project has two purposes:
- provides real examples to use [GBDT-MO](https://github.com/zzd1992/GBDTMO)
- provides codes for reproduction the experiments in our [paper](https://arxiv.org/abs/1909.04373).

## How to use
Suppose you have installed `gbdtmo`. If not, please refer [GBDT-MO](https://github.com/zzd1992/GBDTMO).

Find the path of `gbdtmo.so` and modify `Lib_path` in `cfg.py`.
```python
Lib_path = "path to gbdtmo.so"
```
Import `gbdtmo` and load the shared library file
```python
from gbdtmo import load_lib, GBDTSingle, GBDTMulti
import cfg

LIB = load_lib(cfg.Lib_path)
```
Build a `gbdtmo` instance. You must setup the output dimension.
```python
out_dim = 10
params = {"max_depth": 5, "lr": 0.1}
booster = GBDTMulti(LIB, out_dim=out_dim, params=params)
```
Setup your dataset, train and predict. Data in the first tuple is used for training. Data in the second tuple is used for validation which can be omitted. Items in tuples must be a numpy array.
```python
booster.set_data((x_train, y_train), (x_valid, y_valid))
booster.train(num_rounds)
preds = booster.predict(x_valid)
```
For more information, refer the Python scripts or our [documentation](https://gbdtmo.readthedocs.io).

## Reproduction the experiments
Get the performance of non-sparse `gbdtmo` for a specific dataset via
```
python run_peformance.py dataset gbdtmo
```
Get the running time of non-sparse `gbdtmo` and `gbdtso` for each round via
```
python run_time.py dataset
```
Get the performance of sparse `gbdtmo` for a specific dataset via
```
python run_sparse.py dataset -time 0
```
Get the running time of sparse `gbdtmo` for each rounds via
```
python run_sparse.py dataset -time 1
```
See help of those scripts for more details. Results will be recorded in `log/`. Please refer `test.py` to see how to parse them. We provide datasets `mnist`, `mnist_reg`, `yeast` and `Caltech101`. For `nus-wide`, you should download it into `dataset/` from [here](http://sourceforge.net/projects/mulan/files/datasets/nuswide-cVLADplus.rar) and run `loader.py` to pre-process it.
