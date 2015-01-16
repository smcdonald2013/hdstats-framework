import dataImport
import dataCleaning
import dataAnalysis
import dataVisualization
import numpy as np

##@mainpage hdstats-framework
#@brief Framework for import, cleaning, analysis, and visualization of high-dimensional data.
# Installation: None
#
# Usage: 
#@code{.sh}
#    $ python hdstats-framework.py
# @endcode
#
#The main user interface is a command-line based textual menu interface, with the 
#addition of a simple built-in python interpreter to allow the user to access and 
#interact with underlying classes and variables directly for flexibility.
#
#The included documentation of classes and methods is intended to allow developers
#and others familiar with python to use and extend these methods
#
#The code is divided into four main files, containing UI and analysis functions: 
#dataImport.py, dataCleaning.py, dataAnalysis.py, and dataVisualization.py
#as well, as supporting files for analysis classes and checks.
#


## Basic class to hold all datasets
class Dataset(object):
    def __init__(self, data=None, error=None, variableNameToNumber=None, variableNumberToName=None, independentVariable=np.NaN, dependentVariable=np.NaN):
        self.data=data                                  ## NumPy array containing full dataset
        self.model=None                                 ## Last analysis model applied to dataset (to allow examination in interpreter)
        
        ## Optional:
        self.error=error                                ## Numpy array containing errors for each data point
        self.variableNameToNumber=variableNameToNumber  ## Dictonary mapping from variable names to column numbers 
        self.variableNumberToName=variableNumberToName  ## Dictonary mapping from column numbers to variable names
        self.independentVariable=independentVariable    ## Column number of independent variable
        self.dependentVariable=dependentVariable        ## Column number of dependent variable(s)


## Create dataset object
dataset=Dataset()

## Set global pretty printing options for numpy arrays
np.set_printoptions(precision=3, suppress=True)


while True:
    ## Main menu
    s=raw_input('Select task:\n  1. Import\n  2. Data cleaning\n  3. Analysis\n  4. Visualization\n  5. Enter Python interpreter\n  0. Exit\n')
    if s=='0':
        ## Exit program
        break
    elif s=='1':
        ## Enter import menu
        dataset=dataImport.main(dataset)
    elif s=='2':
        ## Enter data cleaning menu
        if dataset.data==None: 
            print 'Must first import dataset\n'
        else: 
            dataset=dataCleaning.main(dataset)
    elif s=='3':
        ## Enter data analysis menu
        if dataset.data==None: 
            print 'Must first import dataset\n'
        else: 
            dataset=dataAnalysis.main(dataset)
    elif s=='4':
        ## Enter data visualization menu
        if dataset.data==None: 
            print 'Must first import dataset\n'
        else: 
            dataVisualization.main(dataset)
    elif s=='5':
        ## Enter built-in interpreter mode

        print 'Type \'0\' or  \'exit\' to exit interpreter\n\n'
        while (s!='exit') & (s!='0'):
            s=raw_input('In: ')
            if s=='':
                pass
            elif (('=' in s) and not (('==' in s) or ('!=' in s) or ('>=' in s) or ('<=' in s))) or ('import' in s):
                # Catch some common commands that can only be handled by exec()
                try:
                    exec(s)
                except Exception, e:
                    print 'Error: %s' % e
            else:
                # Other statements can be evaluated, so we can print the results to screen
                try:
                    print(eval(s))
                except Exception, e:
                    print 'Error: %s' % e
    else:
        ## Return to main menu in case of unrecognized input
        print 'Input not recognized\n'


