import numpy as np

def main(dataset):

    while True:
        s=raw_input('Select data visualization task:\n  1. Print data to screen\n- 0. Exit\n')
        if s=='0' or s=='': # default
            break
        elif s=='1':
            np.set_printoptions(precision=3, suppress=True)
            print(dataset.data)
            print('\n')


