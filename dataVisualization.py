import numpy as np
import matplotlib.pyplot as plt

def main(dataset):

    while True:
        s=raw_input('Select data visualization task:\n  1. Print data to screen\n  2. Scatter Plot\n- 0. Exit\n')
        if s=='0' or s=='': # default
            break
        elif s=='1':
            print(dataset.data)
            print '\n'
        elif s=='2':
            s1=raw_input('Enter index of variable for x-axis\n')
            s2 = raw_input('Enter index of variable for y-axis\n')
            try: n1=int(s1)
            except: n1=0
            try: n2=int(s2)
            except: n2=1
            if n1>dataset.data.shape[1] or n2>dataset.data.shape[1]:
                print('One or both of indices entered is out of range of data.')
            else:
                plt.scatter(dataset.data[:,n1],dataset.data[:,n2])
                plt.show()


