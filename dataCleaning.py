import numpy as np


def main(dataset):
    while True:
        s=raw_input('Select data cleaning method:\n  1. Remove outliers by standard deviations from the mean\n  2. Remove outliers by percentile\n  3. Replace NaNs\n  4. De-mean dataset\n  5. Normalize variance\n- 0. Exit\n')
        if s=='0' or s=='': # default
            break
        elif s=='1':
            dataset=stdclean(dataset)
        elif s=='2':
            dataset=pctclean(dataset)
        elif s=='3':
            dataset=removeNaNs(dataset)
        elif s=='4':
            dataset=demean(dataset)
        elif s=='5':
            dataset=devar(dataset)
        else:
            print('Input not recognized\n')
    return dataset

def stdclean(dataset):
    s=raw_input('Reject outliers how many standard deviations from the mean (default: 3)\n')
    try: n=float(s)
    except: n=3

    # Remove outliers, column-wise
    for i in range(dataset.data.shape[1]):
        if i!=dataset.independentVariable:
            test = (dataset.data[:,i] > (dataset.data[:,i].mean() + n*dataset.data[:,i].std())) | (dataset.data[:,i] < (dataset.data[:,i].mean() - n*dataset.data[:,i].std()))
            dataset.data[test,i] = np.nan
    return dataset
    

def pctclean(dataset):
    s=raw_input('Reject outliers in what upper and lower percentile (default: 0.5)\n')
    try: n=float(s)
    except: n=0.5

    # Remove outliers, column-wise
    for i in range(dataset.data.shape[1]):
        if i!=dataset.independentVariable:
            test = (dataset.data[:,i] > np.percentile(dataset.data[:,i], 100-n)) | (dataset.data[:,i] < np.percentile(dataset.data[:,i], n))
            dataset.data[test,i] = np.nan
    return dataset

def removeNaNs(dataset):
    s=raw_input('Select:\n-1. Replace NaNs with the variable mean\n 2. Replace NaNs with interpolated values\n 3. Delete samples containing any NaN values\n')
    if s=='1' or s=='': # default
        for i in range(dataset.data.shape[1]):
            dataset.data[np.isnan(dataset.data[:,i]),i]=nanmean(dataset.data[:,i])
    elif s=='2':
        for i in range(dataset.data.shape[1]):
            for j in range(dataset.data.shape[0]) * np.isnan(dataset.data[:,i]):
                dataset.data[j,i] = nanmean(dataset.data[(j-1):(j+1),i])
    elif s=='3':
        test = True;
        for i in range(dataset.data.shape[1]):
            test = test * ~np.isnan(dataset.data[:,i])
        dataset.data = dataset.data[test,:]
    else:
         print('Input not recognized\n')

    return dataset

def demean(dataset):
    #Subtract mean, column-wise
    for i in range(dataset.data.shape[1]):
        if i!=dataset.independentVariable:
            dataset.data[:,i] -= nanmean(dataset.data[:,i])
    return dataset

def devar(dataset):
    # Normalize variance by dividing each column by its standard deviation
    for i in range(dataset.data.shape[1]):
        if (np.std(dataset.data[:,i]) > 0) & (i!=dataset.independentVariable):
            dataset.data[:,i] /= np.std(dataset.data[:,i])
    return dataset

def nanmean(data):
    return np.mean(data[~np.isnan(data)])

