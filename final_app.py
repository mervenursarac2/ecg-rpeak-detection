import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import butter, filtfilt
import gradio as gr

# ==========================================
# 1. VERİ YÜKLEME VE FİLTRELEME
# ==========================================
def load_ecg_record(record_id):
    """PhysioNet veritabanından sinyal ve anotasyonları yükler."""
    record = wfdb.rdrecord(record_id, pn_dir="mitdb")
    ann = wfdb.rdann(record_id, "atr", pn_dir="mitdb")
    return record.p_signal[:, 0], ann.sample, record.fs

def bandpass_filter(signal, fs, low=5, high=15, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, signal)

# ==========================================
# 2. MODEL MİMARİSİ
# ==========================================
class ECGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 5)
        self.conv2 = nn.Conv1d(16, 32, 5)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(32 * 47, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).squeeze()

# ==========================================
# 3. METRİK HESAPLAMA MOTORU
# ==========================================
def evaluate_rpeaks(true_peaks, pred_peaks, fs, tol_ms=50):
    """TP, FP, FN ve Zamanlama Hatasını (MAE) hesaplar."""
    tol = int((tol_ms / 1000) * fs)
    true_peaks = list(true_peaks)
    pred_peaks = list(pred_peaks)

    matched_true = set()
    matched_pred = set()
    errors = []

    for i, p in enumerate(pred_peaks):
        for j, t in enumerate(true_peaks):
            if j in matched_true: continue
            if abs(p - t) <= tol:
                matched_true.add(j)
                matched_pred.add(i)
                errors.append(abs(p - t))
                break

    TP = len(matched_pred)
    FP = len(pred_peaks) - TP
    FN = len(true_peaks) - TP
    
    mae_ms = (np.mean(errors) / fs * 1000) if errors else 0
    return TP, FP, FN, mae_ms

# ==========================================
# 4. ANALİZ VE TAHMİN MANTIĞI
# ==========================================
def get_inference_results(record_id, duration=2500):
    """Sinyal üzerinde model tahmini yapar ve tespit edilen tepeleri döner."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGModel().to(device)
    model.load_state_dict(torch.load("ecg_model_multi.pth", map_location=device))
    model.eval()

    signal, true_r, fs = load_ecg_record(record_id)
    signal = bandpass_filter(signal, fs)
    
    original_signal = signal[:duration]
    true_r_subset = true_r[true_r < duration]

    window_size = 200
    step = 5
    qrs_centers = []

    with torch.no_grad():
        for i in range(0, len(original_signal) - window_size, step):
            w = original_signal[i:i+window_size]
            w = (w - np.mean(w)) / (np.std(w) + 1e-8)
            x = torch.tensor(w).float().unsqueeze(0).unsqueeze(0).to(device)
            if model(x).item() > 0.85:
                qrs_centers.append(i + window_size // 2)

    # R-Peak Seçimi (Refractory Period + Local Max/Min Search)
    detected_r = []
    refractory = int(0.25 * fs)
    for c in qrs_centers:
        if detected_r and c - detected_r[-1] < refractory: continue
        left, right = max(0, c - 40), min(len(original_signal), c + 40)
        r = left + np.argmax(np.abs(original_signal[left:right]))
        detected_r.append(r)

    return original_signal, true_r_subset, detected_r, fs

# ==========================================
# 5. GRADIO FONKSİYONLARI
# ==========================================
def single_record_analysis(record_id):
    try:
        signal, true_r, detected_r, fs = get_inference_results(record_id)
        TP, FP, FN, mae_ms = evaluate_rpeaks(true_r, detected_r, fs)
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        fig = plt.figure(figsize=(15, 6))
        plt.plot(signal, label="EKG Sinyali", alpha=0.5)
        plt.scatter(true_r, signal[true_r], color="green", s=150, label="Gerçek", edgecolors="black", zorder=5)
        plt.scatter(detected_r, signal[detected_r], color="red", marker="x", s=120, label="Model", linewidths=3, zorder=10)
        plt.title(f"Kayıt {record_id} Analizi")
        plt.legend()
        
        report = f"✅ TP: {TP} | ❌ FP: {FP} | 🔍 FN: {FN}\n🎯 Precision: {precision:.2f} | 📈 Recall: {recall:.2f} | ⏱️ MAE: {mae_ms:.2f} ms"
        return fig, report
    except Exception as e:
        return plt.figure(), f"Hata: {e}"

def global_evaluation_report():
    test_records = ['100', '101', '103', '105', '111', '117', '212', '213']
    t_TP, t_FP, t_FN = 0, 0, 0
    maes = []

    yield "⌛ Test seti taranıyor, lütfen bekleyin..."
    
    for rid in test_records:
        try:
            _, true_r, detected_r, fs = get_inference_results(rid)
            TP, FP, FN, mae = evaluate_rpeaks(true_r, detected_r, fs)
            t_TP += TP; t_FP += FP; t_FN += FN
            if mae > 0: maes.append(mae)
        except: continue

    precision = t_TP / (t_TP + t_FP) if (t_TP + t_FP) > 0 else 0
    recall = t_TP / (t_TP + t_FN) if (t_TP + t_FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    final_report = (
        f"===== 🚀 GLOBAL MODEL PERFORMANSI =====\n\n"
        f"🔢 Toplam Test Edilen Kayıt: {len(test_records)}\n"
        f"🎯 Precision (Hassasiyet): %{precision*100:.2f}\n"
        f"🔍 Recall (Duyarlılık): %{recall*100:.2f}\n"
        f"🏆 F1-SCORE (Genel Başarı): %{f1*100:.2f}\n"
        f"⏱️ Ortalama Hata (MAE): {np.mean(maes):.2f} ms\n\n"
        f"Modelimiz, kalp atışlarını %{f1*100:.1f} doğrulukla tespit etmektedir."
    )
    yield final_report

# ==========================================
# 6. GRADIO ARAYÜZÜ (Blocks)
# ==========================================
with gr.Blocks(theme=gr.themes.Soft(), title="EKG AI Analiz") as demo:
    gr.Markdown("# ❤️ AI Tabanlı R-Peak Tespit Sistemi")
    
    with gr.Tab("Tekli Kayıt Analizi"):
        record_input = gr.Textbox(label="Kayıt ID", value="212")
        btn_single = gr.Button("Analiz Et", variant="primary")
        plot_output = gr.Plot()
        text_output = gr.Textbox(label="Kayıt Metrikleri")

    with gr.Tab("Genel Başarı Raporu"):
        gr.Markdown("### Modelin Genel Yeteneği\nTest setindeki tüm kayıtlar taranarak toplu başarı ölçülür.")
        btn_eval = gr.Button("Tüm Test Setini Çalıştır (%80 Eğitim dışı)", variant="stop")
        eval_output = gr.Textbox(label="Global Performans Verileri", lines=10)

    btn_single.click(single_record_analysis, record_input, [plot_output, text_output])
    btn_eval.click(global_evaluation_report, None, eval_output)

if __name__ == "__main__":
    demo.launch(share=True)