import sklearn # scikit-learn
import numpy as np

def main(dataset):
    s=raw_input('Select data analysis task:\n 1. Print data to screen (test)\n')
    if s=='1' or s=='': # default
        np.set_printoptions(precision=3, suppress=True)
        print(dataset.data)
        return dataset
    else:
        print('Input not recognized')
        return dataset




