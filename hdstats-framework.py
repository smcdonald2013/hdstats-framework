import dataImport
import dataCleaning
import dataAnalysis
import dataVisualization

class Dataset(object):
    def __init__(self, data=None, error=None, variableNameToNumber=None, variableNumberToName=None, independentVariable=None, dependentVariable=None):
        self.data=data   # Numpy array containing full dataset
        # Optional:
        self.error=error # Numpy array containing errors for each data point
        self.variableNameToNumber=variableNameToNumber # Dictonary mapping from variable names to column numbers 
        self.variableNumberToName=variableNumberToName # Dictonary mapping from column numbers to variable names
        self.independentVariable=independentVariable # Column number of independent variable
        self.dependentVariable=dependentVariable # Column number of dependent variable(s)


# Create dataset object
dataset=Dataset()


while True:
    s=raw_input('Select task:\n  1. Import\n  2. Data cleaning\n  3. Analysis\n  4. Visualization\n  5. Enter Python interpreter\n  0. Exit\n')
    if s=='0':
        break
    elif s=='1':
        dataset=dataImport.main(dataset)
    elif s=='2':
        dataset=dataCleaning.main(dataset)
    elif s=='3':
        dataset=dataAnalysis.main(dataset)
    elif s=='4':
        dataVisualization.main(dataset)
    elif s=='5':
        print 'Type \'0\' or  \'exit\' to exit interpreter\n\n'
        while (s!='exit') & (s!='0'):
            s=raw_input('In: ')
            if s=='':
                pass
            elif ('=' in s) or ('import' in s):
                try:
                    exec(s)
                except Exception, e:
                    print 'Error: %s' % e
            else:
                try:
                    print(eval(s))
                except Exception, e:
                    print 'Error: %s' % e
    else:
        print 'Input not recognized\n'


