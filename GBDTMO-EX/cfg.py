# please modify it
Lib_path = "../GBDTMO/build/gbdtmo.so"

Depth = {"lightgbm": {"mnist": 6, "yeast": 4, "Caltech101": 8, "nus-wide": 8, "mnist_reg": 8},
         "gbdtso": {"mnist": 6, "yeast": 5, "Caltech101": 8, "nus-wide": 8, "mnist_reg": 8},
         "gbdtmo": {"mnist": 8, "yeast": 6, "Caltech101": 9, "nus-wide": 8, "mnist_reg": 7},
         }

Learning_rate = {"lightgbm": {"mnist": 0.25, "yeast": 0.1, "Caltech101": 0.1, "nus-wide": 0.05, "mnist_reg": 0.1},
                 "gbdtso": {"mnist": 0.1, "yeast": 0.1, "Caltech101": 0.05, "nus-wide": 0.05, "mnist_reg": 0.1},
                 "gbdtmo": {"mnist": 0.1, "yeast": 0.25, "Caltech101": 0.1, "nus-wide": 0.1, "mnist_reg": 0.1},
                }

