import torch
from torch.utils.data import Dataset, DataLoader
from classes import SentimentDataset
import torch.optim as optim
from data import vocab, vocab_size

# Phul luc B: model.py
import torch.nn as nn
import torchtext.vocab as tvocab
import numpy as np

# --- Helper function to load GloVe embeddings ---
def load_glove_embeddings(glove_path, vocab, embedding_dim):
    """
    Loads GloVe embeddings for words found in the vocabulary.

    Args:
        glove_path (str): Name of the GloVe vectors (e.g., 'glove.6B.100d').
                          Make sure you have downloaded these or torchtext can download them.
        vocab (dict): The vocabulary mapping words to indices.
        embedding_dim (int): The dimension of the GloVe embeddings.

    Returns:
        torch.Tensor: The embedding matrix.
    """
    print(f"Loading GloVe vectors: {glove_path}...")
    # Tải GloVe vectors sử dụng torchtext
    # Lần đầu chạy có thể mất thời gian để tải file GloVe
    try:
        glove = tvocab.GloVe(name=glove_path.split('.')[1], # e.g., '6B'
                             dim=embedding_dim,            # e.g., 100
                             cache='.vector_cache')        # Thư mục lưu cache
        print("GloVe vectors loaded successfully.")
    except Exception as e:
        print(f"Error loading GloVe vectors: {e}")
        print("Please ensure the GloVe files are available or can be downloaded.")
        print("You might need to install torchtext: pip install torchtext")
        # Hoặc tải thủ công từ: https://nlp.stanford.edu/projects/glove/
        # và giải nén vào thư mục .vector_cache
        raise e # Dừng chương trình nếu không tải được GloVe

    vocab_size = len(vocab)
    # Khởi tạo ma trận embedding với giá trị ngẫu nhiên nhỏ
    embeddings = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))
    embeddings[vocab['<PAD>']] = np.zeros(embedding_dim) # Vector 0 cho PAD

    # Điền vào ma trận embedding bằng vector GloVe nếu từ có trong GloVe
    loaded_count = 0
    for word, idx in vocab.items():
        if word in glove.stoi: # stoi: string-to-index mapping trong GloVe object

            embeddings[idx] = glove.vectors[glove.stoi[word]].numpy()
            loaded_count += 1
        # else: để lại giá trị khởi tạo ngẫu nhiên (hoặc có thể gán vector <UNK> nếu muốn)

    print(f"Loaded {loaded_count} vectors from GloVe out of {vocab_size} vocab size.")
    return torch.tensor(embeddings, dtype=torch.float)
# --------------------------------------------------

class RNNModel(nn.Module):
    def __init__(self,vocab, vocab_size, embedding_dim, hidden_dim, output_dim,
                 pad_idx, pretrained=False, glove_path='glove.6B.100d'): # Thêm vocab và glove_path
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.padding_idx = pad_idx # Lấy index của PAD token

        # --- Khởi tạo embedding layer ---
        if pretrained:
            print("Using pre-trained GloVe embeddings.")
            # Tải trọng số GloVe
            pretrained_embeddings = load_glove_embeddings(glove_path, vocab, embedding_dim)
            # Tạo lớp Embedding từ trọng số đã tải
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=False, # Cho phép fine-tuning embedding nếu muốn (False)
                padding_idx=self.padding_idx
            )
        else:
            print("Training embeddings from scratch.")
            # Khởi tạo embedding ngẫu nhiên
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=self.padding_idx
            )

        # --- Khởi tạo khối RNN layer ---
        # [Sinh viên bổ sung: dùng nn.RNN với batch_first=True]
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            bidirectional = True,
            num_layers=1, # Giữ đơn giản với 1 lớp RNN
            batch_first=True, # Quan trọng: input/output có dạng (batch, seq, feature)
        )
        self.dropout = nn.Dropout(0.5)
        # --- Khởi tạo tầng Dense để dự đoán 3 nhãn ---
        # [Sinh viên bổ sung: dùng nn.Linear, nhận hidden state từ RNN]
        self.fc = nn.Linear(
            in_features=hidden_dim, # Input là hidden state cuối cùng của RNN
            out_features=output_dim # Output là số lớp cảm xúc (3)
        )

    def forward(self, text):

        # --- Chuyển text thành embedding ---
        embedded = self.embedding(text)
        # --- Đưa qua khối RNN để lẩy hidden state cuối ---
        rnn_output, hidden = self.rnn(embedded)
        last_hidden = hidden[-1] 
        # --- Đưa hidden state qua tầng Dense để dự đoán 3 nhãn ---
        last_hidden = self.dropout(last_hidden)
        predictions = self.fc(last_hidden)
        return predictions

embedding_dim = 100 # Kích thước vector embedding
hidden_dim = 128    # Kích thước lớp ẩn RNN
output_dim = 3
pad_idx = vocab["<PAD>"]
model_glove= RNNModel(vocab,vocab_size, embedding_dim, hidden_dim, output_dim,pad_idx, pretrained=True,glove_path='glove.6B.100d')
model_scratch= RNNModel(vocab,vocab_size, embedding_dim, hidden_dim, output_dim,pad_idx, pretrained=False)