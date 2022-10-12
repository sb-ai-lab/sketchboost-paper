import os
import loader
import numpy as np


def _help_dict(inp, out, bin, path, func):
    return {'inp': inp, 'out': out, 'bin': bin, 'path': path, 'func': func}


class DataLoader:
    def __init__(self):
        self.dataset_reg = {'random_project': _help_dict(4, 8, 256, None, loader.random_project),
                            'friedman1': _help_dict(10, 5, 256, None, loader.friedman1),
                            'mnist_reg': _help_dict(200, 24, 16, 'dataset/mnist.npz', loader.mnist_reg),
                            'nus-wide': _help_dict(128, 81, 64, 'dataset/nus-wide.npz', loader.nus),
                            }

        self.dataset_cls = {'mnist': _help_dict(784, 10, 8, 'dataset/mnist.npz', loader.mnist_cls),
                            'Caltech101': _help_dict(324, 101, 32, 'dataset/Caltech101.npz', loader.Caltech101),
                            'nus-wide': _help_dict(128, 81, 64, 'dataset/nus-wide.npz', loader.nus),
                            'yeast': _help_dict(8, 10, 32, 'dataset/yeast.data', loader.yeast),
                            }

    def get(self, name, reg=True, samples=1000):
        if reg:
            meta = self.dataset_reg[name]
        else:
            meta = self.dataset_cls[name]

        if meta['path'] is None:
            data = meta['func'](samples)
        else:
            data = meta['func'](meta['path'])

        return data, meta


class MultiLabelEvaluate:
    def __init__(self, label):
        '''

        :param label: n by d matrix, not in sparse format
        '''
        self.label = label

    def P_k(self, preds):
        ind = np.argsort(preds, -1)
        ind = ind[:, -5:]
        ind = ind[:, ::-1]

        _row = np.arange(len(self.label))
        out = np.zeros((len(self.label), 5))

        for i in range(5):
            t = ind[:, i]
            out[:, i] = self.label[_row, t]
            if i>0: out[:, i] += out[:, i-1]

        out = np.mean(out, 0)
        out /= np.array([1, 2, 3, 4, 5])

        return out

    def P(self, preds):
        ind = np.argmax(preds, -1)
        _row = np.arange(len(self.label))
        out = self.label[_row, ind]

        return np.mean(out)

    def DCG_k(self, preds):
        pass
