import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

TIME = 'X_Value'
VOLTAGE = 'Voltage - Dev1_ai0'

def parse_data(filename):
    data = pd.read_excel(filename, header=7, usecols=[0, 1])
    return data


def get_fourier_data(data):
    fourier_data = np.fft.fft(data[VOLTAGE])
    fourier_data = np.abs(fourier_data)
    plt.plot(fourier_data, linewidth=0.5, c='red')
    plt.show()

if __name__ == "__main__":
    data = parse_data('../data/test2hz.xlsx')
    get_fourier_data(data)