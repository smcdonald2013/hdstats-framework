import numpy as np


def main(dataset):

    # Import data
    while True:
        s=raw_input('Select:\n  1. Import data array\n  2. Import uncertainty array\t\t\t(optional) \n  3. Import variable names\t\t\t(optional)\n  4. Assign independent variable\t\t(optional)\n  5. Assign dependent variable\t\t\t(optional)\n- 0. Exit\n')
        if s=='0' or s=='': # default 
            break 
        elif s=='1':
            dataset.data=getArray()
        elif s=='2':
            print 'Assumed 2-sigma, same dimensions as data array'
            dataset.error=getArray()
        elif s=='3': 
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
    s=raw_input('Select file type to import numeric array:\n- 1. Delimited ASCII (e.g. .csv)\n  2. Binary\n')
    if s=='1' or s=='': # default
        data=delimited()
    elif s=='2':
        data=binary()
    else:
        print 'Input not recognized\n'
        return None

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

