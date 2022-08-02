import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def blocker(v):
    """
    reblock the data by successively taking the mean of adjacent data points
    """
    l = len(v)
    new_v = np.zeros(l//2)
    for i in range(l//2):
        new_v[i] = 0.5*(v[2*i]+v[2*i+1])
    return new_v

def block_trim(v):
    """
    trim down the length of a dataset to integer power of 2
    """
    l = len(v)
    k = np.floor(np.log2(l))
    e = int(2**k)
    return v[l-e:]

def blocking(v):
    """
    performing the Flyvbjerg-Petersen blocking analysis for evaluating the standard error on a correlated data set
    """
    df = pd.DataFrame(columns = ['blocks', 'mean', 'stdev', 'std_err', 'std_err_err'])
    while len(v) >= 2:
        n = len(v)
        mean = np.mean(v)
        var = v - mean
        sigmasq = sum(var**2)/n
        stddev = np.sqrt(sigmasq)
        stderr = stddev/np.sqrt(n-1)
        stderrerr = stderr*1/np.sqrt(2*(n-1))
        v = blocker(v)
        new_row = [n, mean, stddev, stderr, stderrerr]
        df.loc[len(df)] = new_row
    return df

# load data
qmcdata = np.loadtxt("data.txt")
# perform the blocking analysis
df = blocking(qmcdata)
# now do some plotting
plt.figure()
plt.errorbar(df.index, df['std_err'], yerr=df['std_err_err'])
plt.xlabel("blocking transformations")
plt.ylabel("standard error")
plt.show()