
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torchtext.vocab as tvocab
import numpy as np
from data import vocab as my_vocab, vocab_size as my_vocab_size

# --- Helper function ---
def fetch_glove_vectors(glove_file, token_to_index, emb_dim):
    try:
        glove_vectors = tvocab.GloVe(name=glove_file.split('.')[1], dim=emb_dim, cache='.vector_cache')
        print("GloVe embeddings loaded.")
    except Exception as err:
        print(f"Failed to load GloVe vectors: {err}")
        raise err

    matrix = np.random.uniform(-0.25, 0.25, (len(token_to_index), emb_dim))
    matrix[token_to_index['<PAD>']] = np.zeros(emb_dim)

    matched = 0
    for token, idx in token_to_index.items():
        if token in glove_vectors.stoi:
            matrix[idx] = glove_vectors.vectors[glove_vectors.stoi[token]].numpy()
            matched += 1

    return torch.tensor(matrix, dtype=torch.float)

# --- Custom RNN Model ---
class CustomSentimentRNN(nn.Module):
    def __init__(self, token_vocab, vocab_len, emb_dim, rnn_hidden, n_classes,
                 pad_token_idx, use_pretrained=False, glove_file='glove.6B.100d'):
        super().__init__()
        self.embedding_dim = emb_dim
        self.hidden_dim = rnn_hidden
        self.output_dim = n_classes
        self.vocab_size = vocab_len
        self.padding_idx = pad_token_idx

        if use_pretrained:
            print("Initializing with GloVe.")
            glove_weights = fetch_glove_vectors(glove_file, token_vocab, emb_dim)
            self.embedding = nn.Embedding.from_pretrained(
                glove_weights, freeze=False, padding_idx=self.padding_idx
            )
        else:
            print("Initializing embeddings from scratch.")
            self.embedding = nn.Embedding(
                num_embeddings=vocab_len,
                embedding_dim=emb_dim,
                padding_idx=self.padding_idx
            )

        self.rnn_layer = nn.RNN(
            input_size=emb_dim,
            hidden_size=rnn_hidden,
            bidirectional=True,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(rnn_hidden, n_classes)

    def forward(self, x):
        x_embed = self.embedding(x)
        rnn_out, hidden = self.rnn_layer(x_embed)
        final_hidden = hidden[-1]
        return self.output_layer(final_hidden)

# --- Initialize Models ---
embedding_size = 100
rnn_hidden_size = 128
num_classes = 3
pad_token_id = my_vocab["<PAD>"]

glove_rnn_model = CustomSentimentRNN(
    token_vocab=my_vocab,
    vocab_len=my_vocab_size,
    emb_dim=embedding_size,
    rnn_hidden=rnn_hidden_size,
    n_classes=num_classes,
    pad_token_idx=pad_token_id,
    use_pretrained=True,
    glove_file='glove.6B.100d'
)

scratch_rnn_model = CustomSentimentRNN(
    token_vocab=my_vocab,
    vocab_len=my_vocab_size,
    emb_dim=embedding_size,
    rnn_hidden=rnn_hidden_size,
    n_classes=num_classes,
    pad_token_idx=pad_token_id,
    use_pretrained=False
)
