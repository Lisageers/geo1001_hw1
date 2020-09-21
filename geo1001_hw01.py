#-- GEO1001.2020--hw01
#-- Lisa Geers 
#-- 5351421

import pandas
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as ss
from tabulate import tabulate


def read_csv(datafile):
    """ Reads csv files to a Pandas dataframe. """
    data = pandas.read_csv(datafile, skiprows=[0, 1, 2, 4])
    return data


def mean_statistics_csv(a, b, c, d, e):
    """ Writes mean statistics to a csv file. """
    data_dict = {}

    mean = a.mean()
    data_dict['Mean A'] = mean
    mean = b.mean()
    data_dict['Mean B'] = mean
    mean = c.mean()
    data_dict['Mean C'] = mean
    mean = d.mean()
    data_dict['Mean D'] = mean
    mean = e.mean()
    data_dict['Mean E'] = mean

    std = a.std()
    data_dict['Standard Devation A'] = std
    std = b.std()
    data_dict['Standard Devation B'] = std
    std = c.std()
    data_dict['Standard Devation C'] = std
    std = d.std()
    data_dict['Standard Devation D'] = std
    std = e.std()
    data_dict['Standard Devation E'] = std

    variance = a.var()
    data_dict['Variance A'] = variance
    variance = b.var()
    data_dict['Variance B'] = variance
    variance = c.var()
    data_dict['Variance C'] = variance
    variance = d.var()
    data_dict['Variance D'] = variance
    variance = e.var()
    data_dict['Variance E'] = variance

    dataframe = pandas.DataFrame(data_dict)
    dataframe.to_csv('mean_statistics.csv')


def mean_statistics(dataframe, sensor):
    """ Calculates mean statistics. """
    mean = dataframe.mean()
    std = dataframe.std()
    variance = dataframe.var()
    print(f"Mean of sensor {sensor}: \n{mean} \n\nVariance of sensor {sensor}: \n{variance}, \n\nStandard deviation of sensor {sensor} \n{std}\n")


def histogram(dataframea, dataframeb, dataframec, dataframed, dataframee, bins, xlim, ylim):
    """ Plot histograms for the 5 sensors. """
    fig = plt.figure()   
    fig.suptitle(f"Histograms of all sensors {bins} bins")

    ax1 = fig.add_subplot(231)
    ax1.set_ylabel('Frequency')
    ax1.set_xlabel('Temperature in C')
    ax1.set_title('Sensor A')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    dataframea['Temperature'].hist(bins=bins)
    
    ax2 = fig.add_subplot(232)
    ax2.set_ylabel('Frequency')
    ax2.set_xlabel('Temperature in C')
    ax2.set_title('Sensor B')
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    dataframeb['Temperature'].hist(bins=bins)
    
    ax3 = fig.add_subplot(233)
    ax3.set_ylabel('Frequency')
    ax3.set_xlabel('Temperature in C')
    ax3.set_title('Sensor C')
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)
    dataframec['Temperature'].hist(bins=bins)
    
    ax4 = fig.add_subplot(234)
    ax4.set_ylabel('Frequency')
    ax4.set_xlabel('Temperature in C')
    ax4.set_title('Sensor D')
    ax4.set_xlim(xlim)
    ax4.set_ylim(ylim)
    dataframed['Temperature'].hist(bins=bins)
    
    ax5 = fig.add_subplot(235)
    ax5.set_ylabel('Frequency')
    ax5.set_xlabel('Temperature in C')
    ax5.set_title('Sensor E')
    ax5.set_xlim(xlim)
    ax5.set_ylim(ylim)
    dataframee['Temperature'].hist(bins=bins)
    
    fig.tight_layout()
    plt.show()


def frequency_polygons(a, b, c, d, e):
    """ Plot the frequency polygons for the five sensors. """
    fig, ax = plt.subplots()
    [frequency,bins] = np.histogram(a, bins=50)
    ax.plot(bins[:-1],frequency)

    [frequency,bins] = np.histogram(b, bins=50)
    ax.plot(bins[:-1],frequency)

    [frequency,bins] = np.histogram(c, bins=50)
    ax.plot(bins[:-1],frequency)

    [frequency,bins] = np.histogram(d, bins=50)
    ax.plot(bins[:-1],frequency)

    [frequency,bins] = np.histogram(e, bins=50)
    ax.plot(bins[:-1],frequency)
    
    ax.set(xlabel='Temperature in C', ylabel='Frequency',
       title='Frequency Polygons', xlim=[0, 40], ylim=[0, 250])
    ax.legend(["Temperature sensor A", "Temperature sensor B", "Temperature sensor C", 
        "Temperature sensor D", "Temperature sensor E"])
    plt.show()


def boxplots(a, b, c, d, e, variable, ylim):
    """ Plot boxplots for the five sensors. """
    fig = plt.figure()
    fig.suptitle(f"Boxplots of all sensors {variable}")

    ax1 = fig.add_subplot(231)
    ax1.set_title('Sensor A')
    ax1.set_ylabel(f'{variable}')
    ax1.set_ylim(ylim)
    a.plot.box()
    plt.xticks([])
    
    ax2 = fig.add_subplot(232)
    ax2.set_title('Sensor B')
    ax2.set_ylabel(f'{variable}')
    ax2.set_ylim(ylim)
    b.plot.box()
    plt.xticks([])

    
    ax3 = fig.add_subplot(233)
    ax3.set_title('Sensor C')
    ax3.set_ylabel(f'{variable}')
    ax3.set_ylim(ylim)
    c.plot.box()
    plt.xticks([])
    
    ax4 = fig.add_subplot(234)
    ax4.set_title('Sensor D')
    ax4.set_ylabel(f'{variable}')
    ax4.set_ylim(ylim)
    d.plot.box()
    plt.xticks([])
    
    ax5 = fig.add_subplot(235)
    ax5.set_title('Sensor E')
    ax5.set_ylabel(f'{variable}')
    ax5.set_ylim(ylim)
    e.plot.box()
    plt.xticks([])
    
    fig.tight_layout()
    plt.show()


def pmf(dataframea, dataframeb, dataframec, dataframed, dataframee):
    """ Plot PMF for the five sensors. """
    fig = plt.figure()
    fig.suptitle('Probability Mass Functions of Temperature')

    p = dataframea['Temperature'].value_counts()/len(dataframea['Temperature'])
    sortedp = p.sort_index()
    ax1 = fig.add_subplot(231)
    ax1.bar(sortedp.index,sortedp)
    ax1.set_ylabel("Probability")
    ax1.set_xlabel("Temperature in C")
    ax1.set_title('Sensor A')
    plt.xlim(0, 40)

    p = dataframeb['Temperature'].value_counts()/len(dataframeb['Temperature'])
    sortedp = p.sort_index()
    ax2 = fig.add_subplot(232)
    ax2.bar(sortedp.index,sortedp)
    ax2.set_ylabel("Probability")
    ax2.set_xlabel("Temperature in C")
    ax2.set_title('Sensor B')
    plt.xlim(0, 40)
    plt.ylim(0, 0.025)

    p = dataframec['Temperature'].value_counts()/len(dataframec['Temperature'])
    sortedp = p.sort_index()
    ax3 = fig.add_subplot(233)
    ax3.bar(sortedp.index,sortedp)
    ax3.set_ylabel("Probability")
    ax3.set_xlabel("Temperature in C")
    ax3.set_title('Sensor C')
    plt.xlim(0, 40)
    plt.ylim(0, 0.025)

    p = dataframed['Temperature'].value_counts()/len(dataframed['Temperature'])
    sortedp = p.sort_index()
    ax4 = fig.add_subplot(234)
    ax4.bar(sortedp.index,sortedp)
    ax4.set_ylabel("Probability")
    ax4.set_xlabel("Temperature in C")
    ax4.set_title('Sensor D')
    plt.xlim(0, 40)
    plt.ylim(0, 0.025)

    p = dataframee['Temperature'].value_counts()/len(dataframee['Temperature'])
    sortedp = p.sort_index()
    ax5 = fig.add_subplot(235)
    ax5.bar(sortedp.index,sortedp)
    ax5.set_ylabel("Probability")
    ax5.set_xlabel("Temperature in C")
    ax5.set_title('Sensor E')
    plt.xlim(0, 40)
    plt.ylim(0, 0.025)
    
    fig.tight_layout()
    plt.show()


def pdf(a, b, c, d, e, variable, xlim, ylim):
    """ Plot PDF for the five sensors. """
    fig = plt.figure()
    fig.suptitle(f'Probability Density Functions of {variable}')
    ax1 = fig.add_subplot(231)
    ax1.hist(x=a.astype(float), density=True, bins= 50,alpha=0.7, rwidth=0.85)
    ax1.set_ylabel("Probability density")
    ax1.set_xlabel(f"{variable}")
    ax1.set_title('Sensor A')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)

    ax2 = fig.add_subplot(232)
    ax2.hist(x=b.astype(float), density=True, bins= 50,alpha=0.7, rwidth=0.85)
    ax2.set_title('Sensor B')
    ax2.set_ylabel("Probability density")
    ax2.set_xlabel(f"{variable}")
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)

    ax3 = fig.add_subplot(233)
    ax3.hist(x=c.astype(float), density=True, bins= 50,alpha=0.7, rwidth=0.85)
    ax3.set_title('Sensor C')
    ax3.set_ylabel("Probability density")
    ax3.set_xlabel(f"{variable}")
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)

    ax4 = fig.add_subplot(234)
    ax4.hist(x=d.astype(float), density=True, bins= 50,alpha=0.7, rwidth=0.85)
    ax4.set_title('Sensor D')
    ax4.set_ylabel("Probability density")
    ax4.set_xlabel(f"{variable}")
    ax4.set_xlim(xlim)
    ax4.set_ylim(ylim)

    ax5 = fig.add_subplot(235)
    ax5.hist(x=e.astype(float), density=True, bins= 50,alpha=0.7, rwidth=0.85)
    ax5.set_title('Sensor E')
    ax5.set_ylabel("Probability density")
    ax5.set_xlabel(f"{variable}")
    ax5.set_xlim(xlim)
    ax5.set_ylim(ylim)

    fig.tight_layout()
    plt.show()


def cdf(a, b, c, d, e, variable, xlim):
    """ Plot CDF for the five sensors. """
    fig = plt.figure()
    fig.suptitle(f"Cumulative Density Functions of {variable}")

    ax1 = fig.add_subplot(231)
    ax1.hist(x=a.astype(float), cumulative=True, density=True, bins= 50,alpha=0.7, rwidth=0.85)
    ax1.set_title('Sensor A')
    ax1.set_ylabel("Cumulative probability")
    ax1.set_xlabel(f"{variable}")
    ax1.set_ylim(0, 1)
    ax1.set_xlim(xlim)

    ax2 = fig.add_subplot(232)
    ax2.hist(x=b.astype(float), cumulative=True, density=True, bins= 50,alpha=0.7, rwidth=0.85)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(xlim)
    ax2.set_ylabel("Cumulative probability")
    ax2.set_xlabel(f"{variable}")
    ax2.set_title('Sensor B')

    ax3 = fig.add_subplot(233)
    ax3.hist(x=c.astype(float), cumulative=True, density=True, bins= 50,alpha=0.7, rwidth=0.85)
    ax3.set_ylim(0, 1)
    ax3.set_xlim(xlim)
    ax3.set_ylabel("Cumulative probability")
    ax3.set_xlabel(f"{variable}")
    ax3.set_title('Sensor C')

    ax4 = fig.add_subplot(234)
    ax4.hist(x=d.astype(float), cumulative=True, density=True, bins= 50,alpha=0.7, rwidth=0.85)
    ax4.set_ylim(0, 1)
    ax4.set_xlim(xlim)
    ax4.set_ylabel("Cumulative probability")
    ax4.set_xlabel(f"{variable}")
    ax4.set_title('Sensor D')

    ax5 = fig.add_subplot(235)
    ax5.hist(x=e.astype(float), cumulative=True, density=True, bins= 50,alpha=0.7, rwidth=0.85)
    ax5.set_ylim(0, 1)
    ax5.set_xlim(xlim)
    ax5.set_ylabel("Cumulative probability")
    ax5.set_xlabel(f"{variable}")
    ax5.set_title('Sensor E')

    fig.tight_layout()
    plt.show()


def kernel_dens(a, b, c, d, e):
    """ Plot KDE for the five sensors. """
    fig = plt.figure()
    fig.suptitle(f"Kernel density estimation plots of Wind Speed")
    
    ax1 = fig.add_subplot(231)
    sns.distplot(a.astype(float),ax=ax1, kde=True, hist=False)
    ax1.set_title('Sensor A')
    ax1.set_xlabel('Wind Speed in m/s')
    ax1.set_ylabel("Density")
    ax1.set_xlim([-1, 10])
    ax1.set_ylim([0, 1.5])

    ax2 = fig.add_subplot(232)
    sns.distplot(b.astype(float),ax=ax2, kde=True, hist=False)
    ax2.set_title('Sensor B')
    ax2.set_xlabel('Wind Speed in m/s')
    ax2.set_ylabel("Density")
    ax2.set_xlim([-1, 10])
    ax2.set_ylim([0, 1.5])

    ax3 = fig.add_subplot(233)
    sns.distplot(c.astype(float),ax=ax3, kde=True, hist=False)
    ax3.set_title('Sensor C')
    ax3.set_xlabel('Wind Speed in m/s')
    ax3.set_ylabel("Density")
    ax3.set_xlim([-1, 10])
    ax3.set_ylim([0, 1.5])

    ax4 = fig.add_subplot(234)
    sns.distplot(d.astype(float),ax=ax4, kde=True, hist=False)
    ax4.set_title('Sensor D')
    ax4.set_xlabel('Wind Speed in m/s')
    ax4.set_ylabel("Density")
    ax4.set_xlim([-1, 10])
    ax4.set_ylim([0, 1.5])

    ax5 = fig.add_subplot(235)
    sns.distplot(e.astype(float),ax=ax5, kde=True, hist=False)
    ax5.set_title('Sensor E')
    ax5.set_xlabel('Wind Speed in m/s')
    ax5.set_ylabel("Density")
    ax5.set_xlim([-1, 10])
    ax5.set_ylim([0, 1.5])

    fig.tight_layout()
    plt.show()


def corr(a, b, c, d, e, variable):
    """ Plot Pearson and Spearman correlation for the five sensors. """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title(f'Correlation sensors {variable}')
    ax1.set_xlabel('Pearson correlation coefficient')
    ax1.set_ylabel('Spearman correlation coefficient')

    sp = ss.spearmanr(a, b)[0]
    pe = ss.pearsonr(a, b)[0]
    ax1.scatter(pe, sp)
    plt.annotate('AB', [pe, sp])

    sp = ss.spearmanr(a, c)[0]
    pe = ss.pearsonr(a, c)[0]
    ax1.scatter(pe, sp)
    plt.annotate('AC', [pe, sp])

    sp = ss.spearmanr(a, d)[0]
    pe = ss.pearsonr(a, d)[0]
    ax1.scatter(pe, sp)
    plt.annotate('AD', [pe, sp])

    sp = ss.spearmanr(a, e)[0]
    pe = ss.pearsonr(a, e)[0]
    ax1.scatter(pe, sp)
    plt.annotate('AE', [pe, sp])

    sp = ss.spearmanr(b, c)[0]
    pe = ss.pearsonr(b, c)[0]
    ax1.scatter(pe, sp)
    plt.annotate('BC', [pe, sp])

    sp = ss.spearmanr(b, d)[0]
    pe = ss.pearsonr(b, d)[0]
    ax1.scatter(pe, sp)
    plt.annotate('BD', [pe, sp])

    sp = ss.spearmanr(b, e)[0]
    pe = ss.pearsonr(b, e)[0]
    ax1.scatter(pe, sp)
    plt.annotate('BE', [pe, sp])

    sp = ss.spearmanr(c, d)[0]
    pe = ss.pearsonr(c, d)[0]
    ax1.scatter(pe, sp)
    plt.annotate('CD', [pe, sp])

    sp = ss.spearmanr(c, e)[0]
    pe = ss.pearsonr(c, e)[0]
    ax1.scatter(pe, sp)
    plt.annotate('CE', [pe, sp])

    sp = ss.spearmanr(e, d)[0]
    pe = ss.pearsonr(e, d)[0]
    ax1.scatter(pe, sp)
    plt.annotate('DE', [pe, sp])
    
    plt.show()


def confidence_interval(a, b, c, d, e, variable):
    """ Write confidence intervals to csv file. """
    dict = {}
    ci = ss.t.interval(alpha=0.95, df=len(a)-1, loc=np.mean(a), scale=ss.sem(a)) 
    dict[f"{variable}"] = [f"CI A: {ci}"]

    ci = ss.t.interval(alpha=0.95, df=len(b)-1, loc=np.mean(b), scale=ss.sem(b)) 
    dict[f"{variable}"].append(f"CI B: {ci}")

    ci = ss.t.interval(alpha=0.95, df=len(c)-1, loc=np.mean(c), scale=ss.sem(c)) 
    dict[f"{variable}"].append(f"CI C: {ci}")

    ci = ss.t.interval(alpha=0.95, df=len(d)-1, loc=np.mean(d), scale=ss.sem(d)) 
    dict[f"{variable}"].append(f"CI D: {ci}")

    ci = ss.t.interval(alpha=0.95, df=len(e)-1, loc=np.mean(e), scale=ss.sem(e)) 
    dict[f"{variable}"].append(f"CI E: {ci}")

    f = open("confidence_intervals.txt", "a")
    f.write(tabulate(dict, headers="keys"))
    f.write('\n\n')
    f.close()


def hypothesis_test(a, b, c, d, e, variable):
    """ Calculate hypothesis test for sensor pairs. """
    t, p = ss.ttest_ind(e, d)
    print(f"p-value E, D of {variable}:  {p}")

    t, p = ss.ttest_ind(d, c)
    print(f"p-value C, D of {variable}:  {p}")

    t, p = ss.ttest_ind(b, c)
    print(f"p-value B, C of {variable}:  {p}")

    t, p = ss.ttest_ind(b, a)
    print(f"p-value A, B of {variable}:  {p}")


if __name__ == '__main__':
    data_a = read_csv('data_hw01/HEAT-A_final.csv')
    data_b = read_csv('data_hw01/HEAT-B_final.csv')
    data_c = read_csv('data_hw01/HEAT-C_final.csv')
    data_d = read_csv('data_hw01/HEAT-D_final.csv')
    data_e = read_csv('data_hw01/HEAT-E_final.csv')

    # a1
    mean_statistics(data_a, "A")
    mean_statistics(data_b, "B")
    mean_statistics(data_c, "C")
    mean_statistics(data_d, "D")
    mean_statistics(data_e, "E")

    # mean_statistics_csv(data_a, data_b, data_c, data_d, data_e)

    histogram(data_a, data_b, data_c, data_d, data_e, 5, [0, 40], [0, 1400])
    histogram(data_a, data_b, data_c, data_d, data_e, 50, [0, 40], [0, 250])

    frequency_polygons(data_a['Temperature'], data_b['Temperature'], data_c['Temperature'],
        data_d['Temperature'], data_e['Temperature'])

    boxplots(data_a['Wind Speed'], data_b['Wind Speed'], data_c['Wind Speed'],
        data_d['Wind Speed'], data_e['Wind Speed'], 'Wind Speed in m/s', [-1, 10])

    boxplots(data_a['Direction ‚ True'], data_b['Direction ‚ True'], data_c['Direction ‚ True'],
        data_d['Direction ‚ True'], data_e['Direction ‚ True'], 'Wind Direction in m/s', [-10, 400])
    
    boxplots(data_a['Temperature'], data_b['Temperature'], data_c['Temperature'],
        data_d['Temperature'], data_e['Temperature'], 'Temperature in C', [-1, 40])

    # a2
    pmf(data_a, data_b, data_c, data_d, data_e)

    pdf(data_a['Temperature'], data_b['Temperature'], data_c['Temperature'],
        data_d['Temperature'], data_e['Temperature'], 'Temperature in C', [0, 40], [0, 0.2])

    cdf(data_a['Temperature'], data_b['Temperature'], data_c['Temperature'],
        data_d['Temperature'], data_e['Temperature'], 'Temperature in C', [0, 35])

    pdf(data_a['Wind Speed'], data_b['Wind Speed'], data_c['Wind Speed'],
        data_d['Wind Speed'], data_e['Wind Speed'], 'Wind Speed in m/s', [-1, 10], [0, 5.5])

    kernel_dens(data_a['Wind Speed'], data_b['Wind Speed'], data_c['Wind Speed'],
        data_d['Wind Speed'], data_e['Wind Speed'])

    # a3
    corr(data_a['Temperature'][0:2474], data_b['Temperature'][0:2474], data_c['Temperature'][0:2474],
        data_d['Temperature'][0:2474], data_e['Temperature'][0:2474], 'Temperature')

    corr(data_a['WBGT'][0:2474], data_b['WBGT'][0:2474], data_c['WBGT'][0:2474],
        data_d['WBGT'][0:2474], data_e['WBGT'][0:2474], 'Wet Bulb Globe Temperature')

    corr(data_a['Crosswind Speed'][0:2474], data_b['Crosswind Speed'][0:2474], data_c['Crosswind Speed'][0:2474],
        data_d['Crosswind Speed'][0:2474], data_e['Crosswind Speed'][0:2474], 'Crosswind Speed')

    # a4
    cdf(data_a['Temperature'], data_b['Temperature'], data_c['Temperature'],
        data_d['Temperature'], data_e['Temperature'], 'Temperature in C', [0, 35])

    cdf(data_a['Wind Speed'], data_b['Wind Speed'], data_c['Wind Speed'],
        data_d['Wind Speed'], data_e['Wind Speed'], 'Wind Speed in m/s', [0, 9])

    confidence_interval(data_a['Temperature'], data_b['Temperature'], data_c['Temperature'],
        data_d['Temperature'], data_e['Temperature'], "Temperature")
    
    confidence_interval(data_a['Wind Speed'], data_b['Wind Speed'], data_c['Wind Speed'],
        data_d['Wind Speed'], data_e['Wind Speed'], "Wind Speed")

    hypothesis_test(data_a['Temperature'][0:2474], data_b['Temperature'][0:2474], data_c['Temperature'][0:2474],
         data_d['Temperature'][0:2474], data_e['Temperature'][0:2474], 'Temperature')

    hypothesis_test(data_a['Wind Speed'][0:2474], data_b['Wind Speed'][0:2474], data_c['Wind Speed'][0:2474],
         data_d['Wind Speed'][0:2474], data_e['Wind Speed'][0:2474], 'Wind Speed')