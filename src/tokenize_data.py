import pandas as pd
from transformers import BertTokenizer

# Load data
df = pd.read_csv("data/Fake.csv")
df_real = pd.read_csv("data/True.csv")

# Labels
df['label'] = 0
df_real['label'] = 1

# Combine
df = pd.concat([df, df_real], ignore_index=True)

# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

# Combine text
df['content'] = df['title'] + " " + df['text']

# Take small sample (for faster testing)
df = df.head(1000)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize
encodings = tokenizer(
    df['content'].tolist(),
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors='pt'
)

print("Input IDs shape:", encodings['input_ids'].shape)
print("Attention mask shape:", encodings['attention_mask'].shape)