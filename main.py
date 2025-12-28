from src.load_data import load_ecg_record
from src.filtering import apply_bandpass_filter
from src.dataset import ECGDataset
from src.model import ECGModel
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, ConcatDataset

def main():
    # 1. Kayıt Listesini Belirle (Eklemek istediğin kayıtları buraya yaz)
    # MIT-BIH veri setindeki farklı karakteristiklere sahip bazı kayıtlar:
    record_ids = ['100', '101', '103', '105', '111']
    
    all_datasets = []

    print(f"--- Veri Hazırlama Başladı ({len(record_ids)} kayıt işleniyor) ---")

    for rid in record_ids:
        try:
            # Her kaydı yükle ve filtrele
            signal, r_peaks, fs = load_ecg_record(rid)
            filtered = apply_bandpass_filter(signal, fs=fs)
            
            # Her kayıt için ayrı bir dataset nesnesi oluştur
            current_dataset = ECGDataset(filtered, r_peaks, window_size=200)
            all_datasets.append(current_dataset)
            
            print(f"Kayıt {rid} başarıyla yüklendi. Pencere sayısı: {len(current_dataset)}")
        except Exception as e:
            print(f"Kayıt {rid} yüklenirken hata oluştu: {e}")

    # 2. Tüm Veri Setlerini Birleştir
    # ConcatDataset, farklı kayıtları tek bir dev veri seti gibi birleştirir
    full_dataset = ConcatDataset(all_datasets)
    
    print("-" * 30)
    print(f"Toplam Pencere Sayısı: {len(full_dataset)}")

    # 3. Eğitim ve Test Setlerine Böl (%80 Eğitim, %20 Test)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # 4. DataLoader'ları Hazırla
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Eğitim örneği: {len(train_dataset)}, Test örneği: {len(test_dataset)}")
    print("-" * 30)

    # 5. Model Kontrolü
    model = ECGModel()
    
    # Örnek bir paketi kontrol edelim
    sample_x, sample_y = next(iter(train_loader))
    output = model(sample_x)
    
    print(f"Giriş Paketi Şekli (Batch): {sample_x.shape}") 
    print(f"Model Tahmin Şekli: {output.shape}")
    print(f"İlk 5 Tahmin Olasılığı (Rastgele): {output[:5].detach().numpy()}")
    print("-" * 30)

    # --- EĞİTİM KISMI ---
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 15 # Veri seti büyüdüğü için 15 epoch genellikle yeterlidir
    model.load_state_dict(torch.load('ecg_model_multi.pth'))
    # print(f"\nEğitim Başlıyor... Toplam Epoch: {epochs}")
    
    # for epoch in range(epochs):
    #     model.train()
    #     running_loss = 0.0
    #     correct = 0
    #     total = 0
        
    #     for inputs, labels in train_loader:
    #         optimizer.zero_grad()
            
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
            
    #         loss.backward()
    #         optimizer.step()
            
    #         running_loss += loss.item()
            
    #         # Doğruluk (Acc) Hesapla
    #         preds = (outputs > 0.5).float()
    #         correct += (preds == labels).sum().item()
    #         total += labels.size(0)
            
    #     epoch_loss = running_loss / len(train_loader)
    #     epoch_acc = (correct / total) * 100
        
    #     print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Acc: %{epoch_acc:.2f}")

    # Modeli Kaydet
    torch.save(model.state_dict(), 'ecg_model_multi.pth')
    print("\nEğitim tamamlandı ve model 'ecg_model_multi.pth' adıyla kaydedildi.")

        # --- TEST AŞAMASI ---
    model.eval() # Modeli değerlendirme moduna al (Dropout kapanır)
    test_correct = 0
    test_total = 0

    with torch.no_grad(): # Gradyan hesaplamayı kapat (Hız ve bellek için)
        for inputs, labels in test_loader:
            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    print(f"\n--- TEST SONUCU ---")
    print(f"Hiç görülmemiş verilerde Doğruluk: %{(test_correct / test_total) * 100:.2f}")

    samples, targets = next(iter(test_loader))
    with torch.no_grad():
        preds = model(samples)
    
    plt.figure(figsize=(12, 8))
    for i in range(4): # İlk 4 örneği çizdir
        plt.subplot(4, 1, i+1)
        plt.plot(samples[i][0].numpy())
        plt.title(f"Gerçek: {int(targets[i])} | Tahmin: {preds[i]:.4f}")
        if preds[i] > 0.5:
            plt.axvline(x=100, color='r', linestyle='--', label='R-Peak Tespit Edildi')
    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    main()