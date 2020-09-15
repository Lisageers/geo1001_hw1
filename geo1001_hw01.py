#-- GEO1001.2020--hw01
#-- Lisa Geers 
#-- [YOUR STUDENT NUMBER]

import pandas
import matplotlib.pyplot as plt
import numpy as np



def read_csv(datafile):
    data = pandas.read_csv(datafile, skiprows=[0, 1, 2, 4])
    # print(data)
    return data

def mean_statistics(dataframe):

    mean = dataframe.mean()
    std = dataframe.std()
    variance = dataframe.var()
    return f"mean: \n{mean}, \n variance: \n{variance}, \n standard deviation \n{std}\n"

def histogram(dataframea, dataframeb, dataframec, dataframed, dataframee):
    
    plt.axis([0, 40, 0, 1500])
    plt.subplot(2, 3, 1)
    dataframea['Temperature'].hist(bins=5)
    plt.subplot(2, 3, 2)
    dataframeb['Temperature'].hist(bins=5)
    plt.subplot(2, 3, 3)
    dataframec['Temperature'].hist(bins=5)
    plt.subplot(2, 3, 4)
    dataframed['Temperature'].hist(bins=5)
    plt.subplot(2, 3, 5)
    dataframee['Temperature'].hist(bins=5)
    
    plt.show()

    plt.axis([0, 40, 0, 1500])
    plt.subplot(2, 3, 1)
    dataframea['Temperature'].hist(bins=50)
    plt.subplot(2, 3, 2)
    dataframeb['Temperature'].hist(bins=50)
    plt.subplot(2, 3, 3)
    dataframec['Temperature'].hist(bins=50)
    plt.subplot(2, 3, 4)
    dataframed['Temperature'].hist(bins=50)
    plt.subplot(2, 3, 5)
    dataframee['Temperature'].hist(bins=50)
    
    plt.show()


def frequency_polygons(dataframea, dataframeb, dataframec, dataframed, dataframee):

    fig, ax = plt.subplots()
    xa ,ya  = np.unique(dataframea['Temperature'], return_counts=True)
    ax.plot(xa, ya)
    xb ,yb  = np.unique(dataframeb['Temperature'], return_counts=True)
    ax.plot(xb, yb)
    xc, yc  = np.unique(dataframec['Temperature'], return_counts=True)
    ax.plot(xc, yc)
    xd ,yd  = np.unique(dataframed['Temperature'], return_counts=True)
    ax.plot(xd, yd)
    xe ,ye  = np.unique(dataframee['Temperature'], return_counts=True)
    ax.plot(xe, ye)
    
    ax.set(xlabel='Temperature in C', ylabel='Frequency',
       title='Frequency Polygon')
    ax.legend(["Temperature sensor A", "Temperature sensor B", "Temperature sensor C", 
        "Temperature sensor D", "Temperature sensor E"])
    plt.show()

def boxplots(dataframea, dataframeb, dataframec, dataframed, dataframee):

    fig, ax = plt.subplots()
    plt.subplot(2, 3, 1)
    dataframea.boxplot(column='Temperature')
    plt.subplot(2, 3, 2)
    dataframeb.boxplot(column='Temperature')
    plt.subplot(2, 3, 3)
    dataframec.boxplot(column='Temperature')
    plt.subplot(2, 3, 4)
    dataframed.boxplot(column='Temperature')
    plt.subplot(2, 3, 5)
    dataframee.boxplot(column='Temperature')
    plt.show()

    plt.subplot(2, 3, 1)
    dataframea.boxplot(column='Wind Speed')
    plt.subplot(2, 3, 2)
    dataframeb.boxplot(column='Wind Speed')
    plt.subplot(2, 3, 3)
    dataframec.boxplot(column='Wind Speed')
    plt.subplot(2, 3, 4)
    dataframed.boxplot(column='Wind Speed')
    plt.subplot(2, 3, 5)
    dataframee.boxplot(column='Wind Speed')
    plt.show()

    plt.subplot(2, 3, 1)
    dataframea.boxplot(column='Direction ‚ True')
    plt.subplot(2, 3, 2)
    dataframeb.boxplot(column='Direction ‚ True')
    plt.subplot(2, 3, 3)
    dataframec.boxplot(column='Direction ‚ True')
    plt.subplot(2, 3, 4)
    dataframed.boxplot(column='Direction ‚ True')
    plt.subplot(2, 3, 5)
    dataframee.boxplot(column='Direction ‚ True')
    plt.show()


if __name__ == '__main__':
    data_a = read_csv('data_hw01/HEAT-A_final.csv')
    data_b = read_csv('data_hw01/HEAT-B_final.csv')
    data_c = read_csv('data_hw01/HEAT-C_final.csv')
    data_d = read_csv('data_hw01/HEAT-D_final.csv')
    data_e = read_csv('data_hw01/HEAT-E_final.csv')

    # a1
    mean_statistics(data_a)
    mean_statistics(data_b)
    mean_statistics(data_c)
    mean_statistics(data_d)
    mean_statistics(data_e)

    histogram(data_a, data_b, data_c, data_d, data_e)

    frequency_polygons(data_a, data_b, data_c, data_d, data_e)

    boxplots(data_a, data_b, data_c, data_d, data_e)





