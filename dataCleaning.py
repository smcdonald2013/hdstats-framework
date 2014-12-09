import numpy as np


def main(dataset):
    s=raw_input('Select data cleaning method:\n 1. Standard deviations from the mean\n 2. Percentiles\n')
    if s=='1' or s=='': # default
         return stdclean(dataset)
    if s=='2':
         return pctclean(dataset)
    else:
         print('Input not recognized')
         return dataset

def stdclean(dataset):
    s=raw_input('Reject outliers how many standard deviations from the mean (default: 3)\n')
    try: n=float(s)
    except: n=3

    # Remove outliers, column-wise
    for i in range(dataset.data.shape[1]):
        test = (dataset.data[:,i] > (dataset.data[:,i].mean() + n*dataset.data[:,i].std())) | (dataset.data[:,i] < (dataset.data[:,i].mean() - n*dataset.data[:,i].std()))
        dataset.data[test,i]=np.nan

    return dataset
    

def pctclean(dataset):
    s=raw_input('Reject outliers in what upper and lower percentile (default: 0.5)\n')
    try: n=float(s)
    except: n=0.5

    # Remove outliers, column-wise
    for i in range(dataset.data.shape[1]):
        test = (dataset.data[:,i] > np.percentile(dataset.data[:,i], 100-n)) | (dataset.data[:,i] < np.percentile(dataset.data[:,i], n))

        dataset.data[test,i]=np.nan

    return dataset


