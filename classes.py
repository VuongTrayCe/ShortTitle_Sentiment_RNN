import torch  # <--- THÊM DÒNG NÀY
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts  # Danh sách các văn bản đã mã hóa thành số
        self.labels = labels  # Danh sách nhãn

    def __len__(self):
        return len(self.texts)  # Số mẫu dữ liệu

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)
