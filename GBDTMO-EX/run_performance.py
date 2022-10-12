import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("data", help="which dataset to use",
                    choices=['mnist', 'mnist_reg', 'Caltech101', 'nus-wide', 'yeast', 'student'])
parser.add_argument("model", help="which model to use",
                    choices=['gbdtso', 'gbdtmo', 'lightgbm'])
args = parser.parse_args()

data = args.data
model = args.model

if data == 'mnist':
    command = "python3.6 -u multi_class.py {0} -mode {1} | tee log/{0}_{1}.txt".format(data, model)
    os.system(command)

elif data in ['Caltech101', 'yeast']:
    for seed in range(5):
        command = "python3.6 -u multi_class.py {0} -mode {1} -seed {2} | tee log/{0}_{1}_{2}.txt".\
            format(data, model, seed)
        os.system(command)

elif data == 'mnist_reg':
    command = "python3.6 -u multi_reg.py {0} -mode {1} | tee log/{0}_{1}.txt".format(data, model)
    os.system(command)

elif data == 'student':
    for seed in range(5):
        command = "python3.6 -u multi_reg.py {0} -mode {1} -seed {2} | tee log/{0}_{1}_{2}.txt".\
            format(data, model, seed)
        os.system(command)

elif data == 'nus-wide':
    command = "python3.6 -u multi_label.py {0} -mode {1} | tee log/{0}_{1}.txt".format(data, model)
    os.system(command)

else:
    raise ValueError("Unknown dataset!")


