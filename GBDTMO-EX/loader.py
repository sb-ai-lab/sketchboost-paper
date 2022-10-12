import numpy as np
import os


def normalize(x_train, x_test):
    mean = np.mean(x_train, 0)
    std = np.std(x_train, 0)
    x_train = (x_train - mean) / (std + 1e-4)
    x_test = (x_test - mean) / (std + 1e-4)

    return x_train, x_test


# synthetic
def friedman1(N):
    def func(x):
        return np.sin(np.pi*x[:, 0]*x[:, 1]) + 2 * (x[:, 2] - 0.5)**2 + x[:, 3] + 0.5 * x[:, 4]

    x_train = 2 * np.random.rand(N, 10) - 1.0
    y_train = func(x_train)
    y_train = np.tile(np.expand_dims(y_train, 1), (1, 5))
    y_train += 0.1 * np.random.randn(N, 5)
    x_test = 2 * np.random.rand(N, 10) - 1.0
    y_test = func(x_test)
    y_test = np.tile(np.expand_dims(y_test, 1), (1, 5))
    y_test += 0.1 * np.random.randn(N, 5)

    return x_train, y_train, x_test, y_test


def random_project(N):
    def func(x, w):
        return np.dot(x, w)

    w = 2 * np.random.rand(4, 8) - 1.0
    x_train = 2 * np.random.rand(N, 4) - 1.0
    y_train = func(x_train, w)
    x_test = 2 * np.random.rand(N, 4) - 1.0
    y_test = func(x_test, w)

    return x_train, y_train, x_test, y_test


def mnist_cls(path):
    data = np.load(path)
    x_train, y_train = data['x_train'].astype("float64"), data['y_train'].astype("int32")
    x_test, y_test = data['x_test'].astype("float64"), data['y_test'].astype("int32")
    del data
    
    return x_train, y_train, x_test, y_test


def mnist_reg(path):
    data = np.load(path)
    x_train = data['x_train'].astype("float64")
    x_test = data['x_test'].astype("float64")
    del data

    x_train = np.reshape(x_train, (len(x_train), 28, 28))
    x_train = x_train[:, 4:-4, 4:-4]
    x_train, y_train = x_train[:, :10], x_train[:, 10:]
    y_train = y_train[:, :4, 7:13]
    x_train, y_train = np.reshape(x_train, (len(x_train), -1)), np.reshape(y_train, (len(y_train), -1))
    
    x_test = np.reshape(x_test, (len(x_test), 28, 28))
    x_test = x_test[:, 4:-4, 4:-4]
    x_test, y_test = x_test[:, :10], x_test[:, 10:]
    y_test = y_test[:, :4, 7:13]
    x_test, y_test = np.reshape(x_test, (len(x_test), -1)), np.reshape(y_test, (len(y_test), -1))

    return x_train, y_train, x_test, y_test


def Caltech101(path):
    N = 8677
    data = np.load(path)
    x, y = data['x'], data['y']
    del data

    ind = np.arange(N)
    np.random.shuffle(ind)
    t = int(N * 0.7)
    x, y = x[ind], y[ind]
    x_train, x_test = x[:t], x[t:]
    y_train, y_test = y[:t], y[t:]

    return x_train, y_train, x_test, y_test


def nus_pre(path):
    with open(os.path.join(path, "nus-wide-full-cVLADplus-train.arff"), 'r') as f:
        x_train, y_train = [], []
        line = f.readline()
        while line:
            if '.jpg' in line:
                line = line.strip().split(",")
                feature, label = line[1:129], line[129:]
                feature = [float(_) for _ in feature]
                label = [int(_) for _ in label]
                x_train.append(np.array(feature))
                y_train.append(np.array(label))
            line = f.readline()
        x_train = np.stack(x_train, 0).astype("float64")
        y_train = np.stack(y_train, 0).astype("int32")

    with open(os.path.join(path, "nus-wide-full-cVLADplus-test.arff"), 'r') as f:
        x_test, y_test = [], []
        line = f.readline()
        while line:
            if '.jpg' in line:
                line = line.strip().split(",")
                feature, label = line[1:129], line[129:]
                feature = [float(_) for _ in feature]
                label = [int(_) for _ in label]
                x_test.append(np.array(feature))
                y_test.append(np.array(label))
            line = f.readline()
        x_test = np.stack(x_test, 0).astype("float64")
        y_test = np.stack(y_test, 0).astype("int32")

    np.savez("dataset/nus-wide", x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)


def nus(path):
    data = np.load(path)
    x_train, y_train = data['x_train'], data['y_train'].astype("float64")
    x_test, y_test = data['x_test'], data['y_test'].astype("float64")
    del data

    return x_train, y_train, x_test, y_test


def yeast(path):
    d = {"CYT": 0, "NUC": 1, "MIT": 2, "ME3": 3,
         "ME2": 4, "ME1": 5, "EXC": 6, "VAC": 7,
         "POX": 8, "ERL": 9}

    N = 1484
    with open(path, 'r') as f:
        x = np.zeros((N, 8))
        y = np.zeros(N, 'int32')
        lines = f.readlines()
        for num, line in enumerate(lines):
            line = line.split("YEAST")[-1]
            line = line.strip().split("  ")
            value, label = line[:8], line[-1]
            y[num] = d[label]
            for t, _ in enumerate(value):
                x[num, t] = float(_)

    ind = np.arange(N)
    np.random.shuffle(ind)
    t = int(N * 0.7)
    x, y = x[ind], y[ind]
    x_train, x_test = x[:t], x[t:]
    y_train, y_test = y[:t], y[t:]

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    nus_pre('dataset')
