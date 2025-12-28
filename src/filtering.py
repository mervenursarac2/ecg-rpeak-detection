from scipy.signal import butter, lfilter

def apply_bandpass_filter(data, lowcut=5.0, highcut=15.0, fs=360, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)