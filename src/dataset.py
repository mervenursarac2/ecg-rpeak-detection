import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ECGDataset(Dataset):
    def __init__(self, signal, r_peaks, window_size=200):
        self.window_size = window_size
        self.X, self.y = self._create_segments(signal, r_peaks)
        
        # PyTorch için veriyi (örnek_sayısı, kanal_sayısı, uzunluk) formatına getirelim
        # EKG tek kanal olduğu için kanal_sayısı = 1
        self.X = torch.tensor(self.X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def _create_segments(self, signal, r_peaks): #veriyi 200 örnekli küçük pencerelere böler
        X, y = [], []
        half_win = self.window_size // 2
        
        # Pozitif Örnekler (R Tepeleri)
        for peak in r_peaks:
            if peak > half_win and peak < len(signal) - half_win:
                window = signal[peak - half_win : peak + half_win]
                # Z-score Normalizasyon
                window = (window - np.mean(window)) / (np.std(window) + 1e-8)
                X.append(window)
                y.append(1)
        
        # Negatif Örnekler (Rastgele boş alanlar)
        num_pos = len(y)
        neg_count = 0
        while neg_count < num_pos:
            idx = np.random.randint(half_win, len(signal) - half_win)
            # R tepesine 50 örnekten daha yakın olmasın
            if np.all(np.abs(r_peaks - idx) > 50):
                window = signal[idx - half_win : idx + half_win]
                window = (window - np.mean(window)) / (np.std(window) + 1e-8)
                X.append(window)
                y.append(0)
                neg_count += 1
                
        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]