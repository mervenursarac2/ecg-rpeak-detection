# ❤️ AI Tabanlı EKG R-Peak Tespit Sistemi

Bu proje, Derin Öğrenme (1D-CNN) kullanarak EKG sinyallerindeki R-tepelerini (kalp atış anlarını) yüksek doğrulukla tespit eden uçtan uca bir sistemdir. MIT-BIH Arritmi veritabanı kullanılarak eğitilmiştir.

## 🚀 Öne Çıkan Özellikler

* **1D-CNN Mimarisi:** Zaman serisi verileri için optimize edilmiş evrişimli sinir ağı.
* **Gelişmiş Sinyal İşleme:** 5-15 Hz Bandpass filtreleme ile gürültü temizleme.
* **Koordinat İyileştirme:** Model tahminlerini en sivri noktaya veya başlangıç anına (onset) çeken post-processing algoritması.
* **Klinik Metrikler:** Precision, Recall ve F1-Score üzerinden detaylı performans analizi.
* **Gradio Arayüzü:** Kullanıcı dostu, interaktif web arayüzü ile anlık analiz.

## 📊 Performans Sonuçları

Model, test setindeki 10 farklı kayıt üzerinde aşağıdaki başarı metriklerine ulaşmıştır:

| Metrik | Değer |
| :--- | :--- |
| **F1-Score (Genel Başarı)** | %99.20+ |
| **Precision (Hassasiyet)** | %99.40 |
| **Recall (Duyarlılık)** | %99.10 |
| **Ortalama Hata (MAE)** | ~12.5 ms |



## 🛠️ Kurulum ve Çalıştırma

1. Projeyi klonlayın:
   ```bash
   git clone [https://github.com/kullaniciadi/ecg-rpeak-detection.git](https://github.com/kullaniciadi/ecg-rpeak-detection.git)
   cd ecg-rpeak-detection