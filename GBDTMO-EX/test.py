import numpy as np
from dataset import MultiLabelEvaluate
import os
from loader import nus


def test_cls(data):
    for model in ['gbdtso', 'gbdtmo', 'lightgbm']:
        if data in ["Caltech101", "yeast"]:
            out = 0.0
            for seed in [0, 1, 2, 3, 4]:
                file = "{}_{}_{}.txt".format(data, model, seed)
                if model == "lightgbm":
                    key = "Best score: "
                else:
                    key = "Best score "
                with open(os.path.join("log", file), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if key in line:
                            line = line.split(key)[1]
                            line = line.split(' ')
                            out += float(line[0])
            out /= 5

        elif data in ["mnist"]:
            file = "{}_{}.txt".format(data, model)
            if model == "lightgbm":
                key = "Best score: "
            else:
                key = "Best score "
            with open(os.path.join("log", file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if key in line:
                        line = line.split(key)[1]
                        line = line.split(' ')
                        out = float(line[0])
                        break

        else:
            raise ValueError("Unknown dataset!")

        print("{} on {}:\t{:.5f}".format(model, data, out))


def test_reg(data):
    for model in ['gbdtso', 'gbdtmo', 'lightgbm']:
        if data == "student":
            out = 0.0
            for seed in [0, 1, 2, 3, 4]:
                file = "student_{}_{}.txt".format(model, seed)
                key = "Best score: "
                with open(os.path.join("log", file), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if key in line:
                            line = line.split(key)[1]
                            line = line.split(' ')
                            out += float(line[0])
            out /= 5

        elif data == "mnist_reg":
            file = "mnist_reg_{}.txt".format(model)
            key = "Best score: "
            with open(os.path.join("log", file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if key in line:
                        line = line.split(key)[1]
                        line = line.split(' ')
                        out = float(line[0])
                        break

        else:
            raise ValueError("Unknown dataset!")

        print("{} on {}:\t{:.5f}".format(model, data, out))


def test_nus(path):
    _, _, _, label = nus(path)
    m = MultiLabelEvaluate(label)

    for model in ['gbdtso', 'gbdtmo', 'lightgbm']:
        preds = np.load("result/{}.npy".format(model))
        out = m.P(preds)
        print("{} on nus-wide:\t{:.5f}".format(model, out))


def test_topk(file):
    t, s = [], []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if "score" in line:
                line = line.split(" ")
                s.append(float(line[2]))
            if "time" in line:
                line = line.split(" ")
                t.append(float(line[-1]))

    t, s = np.array(t), np.array(s)
    t = np.reshape(t, (len(t)//2, 2))
    s = np.reshape(s, (len(s)//2, 2))

    return t, s
                     

if __name__ == '__main__':
    test_cls('mnist')
    test_cls('yeast')
    test_cls('Caltech101')
    test_reg('mnist_reg')
