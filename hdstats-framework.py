import dataImport
import dataCleaning
import dataAnalysis

class Dataset(object):
    def __init__(self, data=None, error=None, variableNames=None, independentVariable=None):
        self.data=data   # Numpy array containing full dataset
        self.error=error # Numpy array containing errors for each data point (optional)
        self.variableNames=variableNames # Dictonary containing variable names (optional)
        self.independentVariable=independentVariable # Column number of independent variable (optional)


# Create dataset object
dataset=Dataset()


while True:
    s=raw_input('Select task:\n 1. Import\n 2. Data cleaning\n 3. Analysis\n 0. Exit\n')
    if s=='0':
        break
    elif s=='1':
        dataset=dataImport.main(dataset)
    elif s=='2':
        dataset=dataCleaning.main(dataset)
    elif s=='3':
        dataset=dataAnalysis.main(dataset)

    else:
        print('Input not recognized')


