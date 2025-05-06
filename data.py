import pandas as pd
import torch
import string
import re
import nltk
from torch.utils.data import Dataset, DataLoader
from underthesea import text_normalize
from underthesea import word_tokenize
from sklearn.model_selection import train_test_split
from collections import Counter
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
nltk.download('punkt')
from stopwordsiso import stopwords
# from classes import SentimentDataset

nltk.data.find('tokenizers/punkt')  # Kiểm tra xem đã có chưa
nltk.download('punkt', force=True)
nltk.download('punkt_tab')
vietnamese_stopwords = stopwords(['vi'])
dataset = load_dataset("uit-nlp/vietnamese_students_feedback") # 

# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8573337

data = dataset['train'].to_pandas()
val_df = dataset['validation'].to_pandas()
test_df = dataset['test'].to_pandas()
# Concatenate all DataFrames into one (so data is more objective and balance)
data = pd.concat([data, val_df, test_df], ignore_index=True)
data.rename(columns = {'sentence': 'text', 'sentiment': 'label'}, inplace = True)


data = data.dropna()
data = data.drop_duplicates(subset=['text'], keep='first')

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return torch.tensor(text, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def clean_text(text):
    # Bước 1: loại bỏ khoảng trắng đầu/cuối, thay ký tự xuống dòng bằng khoảng trắng
    text = text.strip().replace("\n", " ")

    # Bước 2: chuyển thường và loại bỏ ký tự không phải chữ/số/khoảng trắng
    text = re.sub(r"[^\w\s]", "", text.lower())

    # Bước 3: xóa khoảng trắng thừa
    text = re.sub(r"\s+", " ", text)

    # Bước 4: giảm lặp ký tự (vd: "aaa" -> "a")
    text = re.sub(r'([a-z]+?)\1+', r'\1', text)

    # Bước 5: token hóa tiếng Việt (chuỗi đã xử lý)
    text = word_tokenize(text, format="text")

    return text
    
data['corpus'] = data['text'].map(lambda text: clean_text(text))

labels = data['label'].tolist()
texts = data['corpus']
tokenized_texts = [
    [word for word in word_tokenize(t) if word not in vietnamese_stopwords]
    for t in texts
]


all_words = [w for txt in tokenized_texts for w in txt]
most_common = Counter(all_words).most_common(4998)
vocab = {'<PAD>': 0, '<UNK>': 1}
for i, (w, _) in enumerate(most_common, 2):
    vocab[w] = i
def to_indices(tokens, max_len):
    idxs = [vocab.get(t, 1) for t in tokens][:max_len]
    return idxs + [0] * (max_len - len(idxs))
    
max_len_text = 20
vocab_size= len(vocab)
text_indices = [to_indices(t, max_len_text) for t in tokenized_texts]
train_texts, test_texts, train_labels, test_labels = train_test_split(text_indices, labels, test_size=0.2, random_state=42)

train_dataset = SentimentDataset(train_texts, train_labels)
test_dataset = SentimentDataset(test_texts, test_labels)
train_loader = DataLoader(train_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)
data_to_save = {
    'train_loader': train_loader,
    'test_loader': test_loader,
    'batch_size': 32,
    'vocab': vocab,
    'vocab_size': len(vocab)
}
save_path = 'sentiment_data_loader.pth'
torch.save(data_to_save, save_path)