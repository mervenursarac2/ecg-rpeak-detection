import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.load_data import load_ecg_record
from src.filtering import apply_bandpass_filter
from src.model import ECGModel

# 1. Modeli Yükle
model = ECGModel()
model.load_state_dict(torch.load('ecg_model_multi.pth'))
model.eval()

def analyze_ecg(record_id):
    try:
        # 2. Veri Hazırlama
        signal, r_peaks_true, fs = load_ecg_record(record_id)
        filtered = apply_bandpass_filter(signal, fs=fs)
        
        duration = 2500 
        test_signal = filtered[:duration]
        true_peaks_in_range = r_peaks_true[r_peaks_true < duration]

        # 3. Kayan Pencere Tahmini
        raw_predictions = []
        window_size = 200
        step_size = 5 
        
        with torch.no_grad():
            for i in range(0, len(test_signal) - window_size, step_size):
                window = test_signal[i : i + window_size]
                window = (window - np.mean(window)) / (np.std(window) + 1e-8)
                input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                output = model(input_tensor).item()
                if output > 0.95:
                    raw_predictions.append((i + (window_size // 2), output))

        # 4. NMS
        detected_peaks = []
        if raw_predictions:
            raw_predictions.sort()
            last_pos, max_prob = raw_predictions[0]
            for pos, prob in raw_predictions[1:]:
                if pos - last_pos < 50:
                    if prob > max_prob:
                        last_pos, max_prob = pos, prob
                else:
                    detected_peaks.append(last_pos)
                    last_pos, max_prob = pos, prob
            detected_peaks.append(last_pos)

        # 5. ONSET TESPİTİ (Atım Başlangıcını Bulma)
        # Sinyalin en dibini değil, o dibe giden dik yamacın başlangıcını bulur
        final_peaks = []
        for p in detected_peaks:
            # Tepeden geriye doğru 40 birimlik bir alanı tara
            start_look = max(0, p - 40)
            region = test_signal[start_look:p]
            
            # Sinyalin türevini (değişim hızını) al
            diff = np.abs(np.diff(region))
            
            # Değişimin (dikleşmenin) başladığı ilk noktayı bul
            # Genelde türevin belirli bir eşiği geçtiği ilk yer "başlangıç" anıdır
            threshold = np.max(diff) * 0.2
            onset_idx = 0
            for idx, val in enumerate(diff):
                if val > threshold:
                    onset_idx = idx
                    break
            
            final_peaks.append(start_look + onset_idx)

        # 6. GRAFİK (Başlangıç Anı Odaklı)
        fig = plt.figure(figsize=(15, 6))
        plt.plot(test_signal, label='EKG Sinyali', color='blue', alpha=0.5)
        
        # Gerçek Tepeler (Yeşil)
        plt.scatter(true_peaks_in_range, test_signal[true_peaks_in_range], 
                    color='green', marker='o', s=150, label='Verideki İşaretli An (Onset)', edgecolors='black', zorder=5)
        
        # Model Tahminleri (Kırmızı - Başlangıca Kaydırılmış)
        if final_peaks:
            plt.scatter(final_peaks, test_signal[final_peaks], 
                        color='red', marker='x', s=120, label='Modelin Bulduğu Başlangıç', linewidths=3, zorder=10)

        plt.title(f"Atım Başlangıcı (Onset) Analizi: Kayıt {record_id}", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.2)
        
        return fig, f"Analiz tamamlandı. Atım başlangıçları işaretlendi."

    except Exception as e:
        return plt.figure(), f"Hata: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# 🏥 Kalp Atım Anı (Onset) Dedektörü")
    with gr.Row():
        inp = gr.Textbox(label="Kayıt ID", value="212")
        btn = gr.Button("Analiz Et")
    plot = gr.Plot()
    msg = gr.Textbox(label="Durum")
    btn.click(fn=analyze_ecg, inputs=inp, outputs=[plot, msg])

if __name__ == "__main__":
    demo.launch()