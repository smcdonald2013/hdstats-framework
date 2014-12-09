import numpy as np

class Dataset(object):
    def __init__(self, data, error=None, variableNames=None, independentVariable=None):
        self.data=data   # Numpy array containing full dataset
        self.error=error # Numpy array containing errors for each data point (optional)
        self.variableNames=variableNames # Dictonary containing variable names (optional)
        self.independentVariable=independentVariable # Column number of independent variable (optional) 


def main():
    s=raw_input('Select file type to input:\n 1. Delimited ASCII (e.g. .csv)\n')
    if s=='1' or s=='': # default
        return delimited()
    else:
        print('Input not recognized\n')
        return 0 


def delimited():
    filename=raw_input('Name of delimited file?\n')
    delim=raw_input('Delimiter?\n')

    data=np.genfromtxt(filename, delimiter=delim, dtype=float)

    # Store variables in columns by default
    while True:
        variablesAsColumns=raw_input('Are variables stored as rows (0) or columns (1, default)?\n')
        if variablesAsColumns=='1' or variablesAsColumns=='':
            break
        elif variablesAsColumns=='0':
            data=data.transpose()
            break
        else:
            print('Input not recognized\n')
          
    
    return Dataset(data)

