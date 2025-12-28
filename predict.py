import torch
import numpy as np
import matplotlib.pyplot as plt
from src.load_data import load_ecg_record
from src.filtering import apply_bandpass_filter
from src.model import ECGModel

def predict_and_visualize(record_id='212'):
    # 1. Modeli Yükle
    model = ECGModel()
    # Modelin ağırlıklarını dosyadan oku
    model.load_state_dict(torch.load('ecg_model_multi.pth'))
    model.eval()

    # 2. Hiç Görülmemiş Veriyi Yükle ve Filtrele
    signal, r_peaks_true, fs = load_ecg_record(record_id)
    filtered = apply_bandpass_filter(signal, fs=fs)
    
    # Sinyalin sadece belirli bir kısmını görselleştirelim (İlk 2000 örnek)
    duration = 2000 
    test_signal = filtered[:duration]
    true_peaks_in_range = r_peaks_true[r_peaks_true < duration]

    # 3. Kayan Pencere ile Tahmin ve Temizleme
    raw_predictions = []
    window_size = 200
    step_size = 5 # Daha hassas tarama için adımı küçülttük

    print(f"--- {record_id} Kaydı Analiz Ediliyor ---")
    
    with torch.no_grad():
        for i in range(0, len(test_signal) - window_size, step_size):
            window = test_signal[i : i + window_size]
            
            # Eğitimdeki normalizasyonun aynısı
            window = (window - np.mean(window)) / (np.std(window) + 1e-8)
            
            input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            output = model(input_tensor).item()
            
            # Model %95'ten fazla eminse listeye ekle
            if output > 0.95:
                # (Konum, Olasılık Skoru) olarak kaydet
                raw_predictions.append((i + (window_size // 2), output))

    # --- NMS (Non-Maximum Suppression) Bölümü ---
    # Birbirine çok yakın tahminlerden sadece en iyisini seçer
    detected_peaks = []
    if raw_predictions:
        raw_predictions.sort() # Konuma göre sırala
        
        last_pos, max_prob = raw_predictions[0]
        for pos, prob in raw_predictions[1:]:
            # Eğer yeni tahmin, son bulunan tepeden 50 birimden daha yakınsa
            if pos - last_pos < 50: 
                # Hangisinin olasılığı daha yüksekse onu tut
                if prob > max_prob:
                    last_pos = pos
                    max_prob = prob
            else:
                # Mesafe yeterliyse, önceki en iyi tahmini 'kesin tepe' olarak ekle
                detected_peaks.append(last_pos)
                last_pos = pos
                max_prob = prob
        detected_peaks.append(last_pos) # Son grubu ekle

    # 4. Görselleştirme
    plt.figure(figsize=(15, 6))
    plt.plot(test_signal, label='EKG Sinyali', color='blue', alpha=0.5, linewidth=1)
    
    # Gerçek Tepeler (Yeşil Daireler)
    plt.scatter(true_peaks_in_range, test_signal[true_peaks_in_range], 
                color='green', marker='o', s=150, label='Gerçek R-Tepeleri (Ground Truth)', edgecolors='black')
    
    # Modelin Filtrelenmiş Tahminleri (Kırmızı Çarpılar)
    if detected_peaks:
        plt.scatter(detected_peaks, test_signal[detected_peaks], 
                    color='red', marker='x', s=100, label='Modelin Net Kararı', linewidths=2)

    plt.title(f"Modelin Tanımadığı Kayıt Üzerinde Test: {record_id}")
    plt.xlabel("Örnek Sayısı (Sample)")
    plt.ylabel("Genlik (Amplitude)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"Analiz Tamamlandı. Gerçek Tepe: {len(true_peaks_in_range)}, Bulunan Tepe: {len(detected_peaks)}")

if __name__ == "__main__":
    predict_and_visualize('212')