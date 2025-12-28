from src.load_data import load_ecg_record
from src.filtering import apply_bandpass_filter
from src.visualize import plot_signals

def main():
    # 1. Yükle
    signal, r_peaks, fs = load_ecg_record('100')
    
    # 2. Filtrele
    filtered = apply_bandpass_filter(signal, fs=fs)
    
    # 3. Görselleştir
    plot_signals(signal, filtered, r_peaks, fs)
    
    print("success")
if __name__ == "__main__":
    main()