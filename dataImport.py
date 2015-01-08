import numpy as np


def main(dataset):

    # Import data
    while True:
        s=raw_input('Select:\n  1. Import data array\n  2. Import variable names\t\t\t(optional)\n  3. Set uncertainty array\t\t\t(optional)\n  4. Assign independent variable\t\t(optional)\n  5. Assign dependent variable\t\t\t(optional)\n- 0. Exit\n')
        if s=='0' or s=='': # default 
            break 
        elif s=='1':
            dataset.data=getArray()
            dataset.error=None # Importing a new dataset resets uncertainties
        elif s=='2': 
            # Make dictionaries mapping from variable names to column numbers, and vice versa
            filename=raw_input('Name of delimited ASCII file containing variable names?\n')
            fp=open(filename)
            firstline=fp.readline()
            firstline=firstline.replace('\n','')
            delim=raw_input('Delimiter?\n')
            firstline=firstline.split(delim)
            dataset.variableNameToNumber=dict((firstline[i],i) for i in range(len(firstline)))
            dataset.variableNumberToName={y: x for x, y in dataset.variableNameToNumber.items()}
            fp.close()
        elif s=='3':
            # Import uncertainty array
            if dataset.data==None:
                print 'Error: must import data first'
            else:
                s1=raw_input('Type of uncertainty:\n- 1. Set global relative uncertainty\n  2. Import relative uncertainty matrix\n  3. Import absolute uncertainty matrix\n')
                if s1=='1' or s1=='':
                    s2=raw_input('Enter global relative 2-sigma (percent) uncertainty (default: 5%)\n')
                    try: n = float(s2)
                    except: n = 5
                    dataset.error = dataset.data * n / 100
                elif s1=='2':
                    print 'Assumed 2-sigma (percent), same dimensions as data array\n'
                    dataset.error = getArray()
                    dataset.error = dataset.data * dataset.error / 100
                elif s1=='3':
                    print 'Assumed 2-sigma, same dimensions as data array\n'
                    dataset.error = getArray()
        elif s=='4':
            # Assign independent Variable
            n=None
            s=raw_input('Enter column number of independent variable (zero-indexed)\n')
            try: 
                n=int(s)
            except: 
                print 'Input not an interger. Set to NaN'
                n=np.NaN
            dataset.independentVariable=n
        elif s=='4':
            # Assign dependent Variable
            n=None
            s=raw_input('Enter column number of independent variable (zero-indexed)\n')
            try:
                n=int(s)
            except: 
                print 'Input not an interger. Set to NaN'
                n=np.NaN
            dataset.dependentVariable=n

    return dataset



def getArray():
    # Import numeric array
    data=None
    while data==None:
        s=raw_input('Select file type to import numeric array:\n- 1. Delimited ASCII (e.g. .csv)\n  2. Binary\n')
        if s=='1' or s=='': # default
            data=delimited()
        elif s=='2':
            data=binary()
        else:
            print 'Input not recognized\n'

    # Store variables in columns by default
    variablesAsColumns=raw_input('Are variables stored as:\n  0. rows\n- 1. columns\n')
    if variablesAsColumns=='0':
        data=data.transpose()
    elif not (variablesAsColumns=='1' or variablesAsColumns==''):
        print 'Input not recognized\n'
    return data



def delimited():
    filename=raw_input('Name of delimited file?\n')
    delim=raw_input('Delimiter?\n')
    data=None
    try: data=np.genfromtxt(filename, delimiter=delim, dtype=float)
    except: print '%s: Invalid filename or delimiter\n' % filename
    return data

def binary():
    filename=raw_input('Name of binary file?\n')
    datatype=raw_input('Data type? (e.g. float)\n')
    data=None
    try: data=np.fromfile(filename, dtype=datatype, count=-1, sep='')
    except: print '%s: Invalid file name or data type\n' % filename
    return data

