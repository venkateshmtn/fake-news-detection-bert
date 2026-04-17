import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader

# Load data
df_fake = pd.read_csv("data/Fake.csv")
df_real = pd.read_csv("data/True.csv")

df_fake['label'] = 0
df_real['label'] = 1

df = pd.concat([df_fake, df_real], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)

df['content'] = df['title'] + " " + df['text']

# Use small sample (for speed)
df = df.head(2000)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Custom Dataset
class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=64
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Dataset
dataset = NewsDataset(df['content'].tolist(), df['label'].tolist())

# DataLoader
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

print("Dataset size:", len(df))

for epoch in range(3):
    for i, batch in enumerate(loader):
        print(f"Running batch {i}")   # 👈 ADD THIS

        optimizer.zero_grad()

        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(f"Batch {i}, Loss: {loss.item()}")

import os

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Save model
model.save_pretrained("models/bert_model")
tokenizer.save_pretrained("models/bert_model")

print("✅ Full model saved!")

