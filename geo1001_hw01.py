#-- GEO1001.2020--hw01
#-- Lisa Geers 
#-- [YOUR STUDENT NUMBER]

import pandas
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as ss



def read_csv(datafile):
    data = pandas.read_csv(datafile, skiprows=[0, 1, 2, 4])
    # print(data)
    return data

def mean_statistics(dataframe):

    mean = dataframe.mean()
    std = dataframe.std()
    variance = dataframe.var()
    return print(f"mean: \n{mean}, \n variance: \n{variance}, \n standard deviation \n{std}\n")

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


def pmf(dataframea, dataframeb, dataframec, dataframed, dataframee):

    fig = plt.figure()

    p = dataframea['Temperature'].value_counts()/len(dataframea['Temperature'])
    sortedp = p.sort_index()
    ax1 = fig.add_subplot(231)
    ax1.bar(sortedp.index,sortedp)
    plt.xlim(0, 40)

    p = dataframeb['Temperature'].value_counts()/len(dataframeb['Temperature'])
    sortedp = p.sort_index()
    ax2 = fig.add_subplot(232)
    ax2.bar(sortedp.index,sortedp)
    plt.xlim(0, 40)
    plt.ylim(0, 0.025)

    p = dataframec['Temperature'].value_counts()/len(dataframec['Temperature'])
    sortedp = p.sort_index()
    ax3 = fig.add_subplot(233)
    ax3.bar(sortedp.index,sortedp)
    plt.xlim(0, 40)
    plt.ylim(0, 0.025)

    p = dataframed['Temperature'].value_counts()/len(dataframed['Temperature'])
    sortedp = p.sort_index()
    ax4 = fig.add_subplot(234)
    ax4.bar(sortedp.index,sortedp)
    plt.xlim(0, 40)
    plt.ylim(0, 0.025)

    p = dataframee['Temperature'].value_counts()/len(dataframee['Temperature'])
    sortedp = p.sort_index()
    ax5 = fig.add_subplot(235)
    ax5.bar(sortedp.index,sortedp)
    plt.xlim(0, 40)
    plt.ylim(0, 0.025)
    
    plt.show()


def pdf(a, b, c, d, e):

    fig = plt.figure()
    # fig.set_title('PDF Temperature sensors')
    ax1 = fig.add_subplot(231)
    ax1.hist(x=a.astype(float), density=True, bins= 50,alpha=0.7, rwidth=0.85)
    ax1.set_title('Sensor 1')
    # plt.ylim(0, 0.20)

    ax2 = fig.add_subplot(232)
    ax2.hist(x=b.astype(float), density=True, bins= 50,alpha=0.7, rwidth=0.85)
    # plt.ylim(0, 0.20)
    ax2.set_title('Sensor 2')

    ax3 = fig.add_subplot(233)
    ax3.hist(x=c.astype(float), density=True, bins= 50,alpha=0.7, rwidth=0.85)
    # plt.ylim(0, 0.20)
    ax3.set_title('Sensor 3')

    ax4 = fig.add_subplot(234)
    ax4.hist(x=a.astype(float), density=True, bins= 50,alpha=0.7, rwidth=0.85)
    # plt.ylim(0, 0.20)
    ax4.set_title('Sensor 4')

    ax5 = fig.add_subplot(235)
    ax5.hist(x=a.astype(float), density=True, bins= 50,alpha=0.7, rwidth=0.85)
    # plt.ylim(0, 0.20)
    ax5.set_title('Sensor 5')

    plt.show()


def cdf(a, b, c, d, e):

    fig = plt.figure()
    ax1 = fig.add_subplot(231)
    ax1.hist(x=a.astype(float), cumulative=True, density=True, bins= 50,alpha=0.7, rwidth=0.85)
    ax1.set_title('Sensor 1')
    plt.ylim(0, 1)

    ax2 = fig.add_subplot(232)
    ax2.hist(x=b.astype(float), cumulative=True, density=True, bins= 50,alpha=0.7, rwidth=0.85)
    plt.ylim(0, 1)
    ax2.set_title('Sensor 2')

    ax3 = fig.add_subplot(233)
    ax3.hist(x=c.astype(float), cumulative=True, density=True, bins= 50,alpha=0.7, rwidth=0.85)
    plt.ylim(0, 1)
    ax3.set_title('Sensor 3')

    ax4 = fig.add_subplot(234)
    ax4.hist(x=a.astype(float), cumulative=True, density=True, bins= 50,alpha=0.7, rwidth=0.85)
    plt.ylim(0, 1)
    ax4.set_title('Sensor 4')

    ax5 = fig.add_subplot(235)
    ax5.hist(x=a.astype(float), cumulative=True, density=True, bins= 50,alpha=0.7, rwidth=0.85)
    plt.ylim(0, 1)
    ax5.set_title('Sensor 5')

    plt.show()

def kernel_dens(a, b, c, d, e):
    fig = plt.figure(figsize=(17,6))
    
    ax1 = fig.add_subplot(231)
    sns.distplot(a.astype(float),ax=ax1, kde=True)
    ax1.set_title('Sensor A')
    ax1.set_xlabel('')

    ax2 = fig.add_subplot(232)
    sns.distplot(b.astype(float),ax=ax2, kde=True)
    ax2.set_title('Sensor B')
    ax2.set_xlabel('')

    ax3 = fig.add_subplot(233)
    sns.distplot(c.astype(float),ax=ax3, kde=True)
    ax3.set_title('Sensor C')
    ax3.set_xlabel('')

    ax4 = fig.add_subplot(234)
    sns.distplot(d.astype(float),ax=ax4, kde=True)
    ax4.set_title('Sensor D')
    ax4.set_xlabel('')

    ax5 = fig.add_subplot(235)
    sns.distplot(e.astype(float),ax=ax5, kde=True)
    ax5.set_title('Sensor E')
    ax5.set_xlabel('')

    plt.show()


if __name__ == '__main__':
    data_a = read_csv('data_hw01/HEAT-A_final.csv')
    data_b = read_csv('data_hw01/HEAT-B_final.csv')
    data_c = read_csv('data_hw01/HEAT-C_final.csv')
    data_d = read_csv('data_hw01/HEAT-D_final.csv')
    data_e = read_csv('data_hw01/HEAT-E_final.csv')

    # a1
    # mean_statistics(data_a)
    # mean_statistics(data_b)
    # mean_statistics(data_c)
    # mean_statistics(data_d)
    # mean_statistics(data_e)

    # histogram(data_a, data_b, data_c, data_d, data_e)

    # frequency_polygons(data_a, data_b, data_c, data_d, data_e)

    # boxplots(data_a, data_b, data_c, data_d, data_e)

    # a2

    # pmf(data_a, data_b, data_c, data_d, data_e)

    # pdf(data_a['Temperature'], data_b['Temperature'], data_c['Temperature'],
    #     data_d['Temperature'], data_e['Temperature'])

    # cdf(data_a['Temperature'], data_b['Temperature'], data_c['Temperature'],
    #     data_d['Temperature'], data_e['Temperature'])

    # pdf(data_a['Wind Speed'], data_b['Wind Speed'], data_c['Wind Speed'],
    #     data_d['Wind Speed'], data_e['Wind Speed'])

    kernel_dens(data_a['Wind Speed'], data_b['Wind Speed'], data_c['Wind Speed'],
        data_d['Wind Speed'], data_e['Wind Speed'])
    



