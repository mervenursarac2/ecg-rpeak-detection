import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# 1.1 & 1.4: VERİ VE ANOTASYONLARI OKUMA
# '100' numaralı kaydı okuyoruz
record_path = 'mit-bih-arrhythmia-database-1.0.0/100' 
record = wfdb.rdrecord(record_path)
annotation = wfdb.rdann(record_path, 'atr')

# Tek kanal (Lead II) sinyali alalım
signal = record.p_signal[:, 0] 
fs = record.fs  # Örnekleme frekansı (MIT-BIH için genelde 360Hz)
r_peaks_gt = annotation.sample # Gerçek R-tepesi indeksleri

# 1.3: FİLTRELEME (Bandpass Filter 5-15 Hz)
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

# QRS kompleksini belirginleştirmek için filtreyi uygula
filtered_signal = butter_bandpass_filter(signal, 5.0, 15.0, fs, order=3)

# 1.2: GÖRSELLEŞTİRME
plt.figure(figsize=(15, 8))

# İlk 2000 örneği görelim (yaklaşık 5.5 saniye)
limit = 2000
time_axis = np.arange(limit) / fs

plt.subplot(2, 1, 1)
plt.plot(time_axis, signal[:limit], color='gray', alpha=0.5, label='Ham Sinyal')
plt.title('Ham EKG Sinyali ve Gerçek R-Tepeleri')
plt.ylabel('Genlik (mV)')

# Gerçek R-tepelerini ham sinyal üzerine işaretle
relevant_peaks = r_peaks_gt[r_peaks_gt < limit]
plt.scatter(relevant_peaks / fs, signal[relevant_peaks], color='red', marker='v', label='Gerçek R-Tepesi')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time_axis, filtered_signal[:limit], color='blue', label='Filtrelenmiş Sinyal (5-15 Hz)')
plt.title('Bandpass Filtre Sonrası (QRS Daha Belirgin)')
plt.xlabel('Zaman (saniye)')
plt.ylabel('Genlik')
plt.legend()

plt.tight_layout()
plt.show()