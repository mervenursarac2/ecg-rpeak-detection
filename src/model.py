import torch
import torch.nn as nn
import torch.nn.functional as F

class ECGModel(nn.Module):
    def __init__(self):
        super(ECGModel, self).__init__()
        
        # 1. Evrişim Katmanı: Sinyaldeki temel çizgileri ve ani çıkışları yakalar
        # Input: (Batch, 1, 200) -> Output: (Batch, 16, 196)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5)
        
        # 2. Evrişim Katmanı: Daha karmaşık şekilleri (QRS kompleksinin bütünü gibi) tanır
        # Output: (Batch, 32, 94) (Pooling sonrası boyut küçülecek)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5)
        
        # Max Pooling: Sinyali özetler, en baskın özellikleri tutar
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Dropout: Modelin ezberlemesini (overfitting) önlemek için rastgele nöronları kapatır
        self.dropout = nn.Dropout(0.2)
        
        # Tam Bağlantılı Katmanlar (Fully Connected): Karar verme aşaması
        # Boyut hesaplaması: Katmanlardan geçen sinyalin son halini vektöre çeviriyoruz
        self.fc1 = nn.Linear(32 * 47, 64) 
        self.fc2 = nn.Linear(64, 1) # Tek bir çıkış: 0 ile 1 arasında olasılık
        
    def forward(self, x):
        # Katman 1 + Aktivasyon + Pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Katman 2 + Aktivasyon + Pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Veriyi düzleştir (Flatten) -> FC katmanına hazırlık
        x = x.view(x.size(0), -1) 
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Sonuç: Sigmoid ile 0-1 arasına sıkıştırılmış olasılık
        x = torch.sigmoid(self.fc2(x))
        
        return x.squeeze() # (Batch, 1) -> (Batch) formatına getir