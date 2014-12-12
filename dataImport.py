import numpy as np


def main(dataset):

    # Import data
    while True:
        s=raw_input('Select:\n 1. Import data array\n 2. Import uncertainty array (assumed 2-sigma, same dimensions as data array) \n 3. Import variable names\n 4. Assign independent variable\n 0. Done importing\n')
        
        if s=='0' or s=='': 
            break 
        elif s=='1':
            dataset.data=getArray()
        elif s=='2':
            dataset.error=getArray()
        elif s=='3': 
            # Make a dictionary containing the list of variable names
            filename=raw_input('Name of delimited ASCII file containing variable names?\n')
            fp=open(filename)
            firstline=fp.readline()
            firstline=firstline.replace('\n','')
            delim=raw_input('Delimiter?\n')
            firstline.split(delim) 
            dataset.variableNames=dict((firstline[i],i) for i in range(len(firstline)))
        elif s=='4':
            # Assign independent Variable
            n=None
            s=raw_input('Enter column number of independent variable (zero-indexed)\n')
            try: n=int(s)
            except: print('Input not an integer')
            dataset.independentVariable=n

    return dataset



def getArray():
    # Import numeric array
    s=raw_input('Select file type to import numeric array:\n 1. Delimited ASCII (e.g. .csv)\n')
    if s=='1' or s=='': # default
        data=delimited()
    else:
        print('Input not recognized\n')
        return None

    # Store variables in columns by default
    variablesAsColumns=raw_input('Are variables stored as rows (0) or columns (1, default)?\n')
    if variablesAsColumns=='0':
        data=data.transpose()
    elif not (variablesAsColumns=='1' or variablesAsColumns==''):
        print('Input not recognized\n')
    return data



def delimited():
    filename=raw_input('Name of delimited file?\n')
    delim=raw_input('Delimiter?\n')
    data=None
    try: data=np.genfromtxt(filename, delimiter=delim, dtype=float)
    except: print('%s not found\n', filename)
    return data



