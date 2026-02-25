#pip install torch pandas scikit-learn konlpy
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter
import re

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=50):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def encode(self, text):
        tokens = text.split()
        ids = [self.vocab.get(t, self.vocab["<unk>"]) for t in tokens]
        ids = ids[:self.max_len]
        ids += [self.vocab["<pad>"]] * (self.max_len - len(ids))
        return torch.tensor(ids)

    def __getitem__(self, idx):
        return self.encode(self.texts[idx]), torch.tensor(self.labels[idx])