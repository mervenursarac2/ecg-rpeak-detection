import matplotlib.pyplot as plt
import numpy as np

def plot_signals(raw_signal, filtered_signal, r_peaks, fs, limit=2000):
    time_axis = np.arange(limit) / fs
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, raw_signal[:limit], label='Ham')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, filtered_signal[:limit], label='Filtrelenmiş', color='orange')
    plt.legend()
    plt.show()