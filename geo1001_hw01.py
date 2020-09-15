import pandas


def read_csv(datafile):
    data = pandas.read_csv(datafile)
    print(data)
    return data


if __name__ == '__main__':
    data_a = read_csv('data_hw01/HEAT-A_final.csv')
    data_b = read_csv('data_hw01/HEAT-B_final.csv')
    data_c = read_csv('data_hw01/HEAT-C_final.csv')
    data_d = read_csv('data_hw01/HEAT-D_final.csv')
    data_e = read_csv('data_hw01/HEAT-E_final.csv')
