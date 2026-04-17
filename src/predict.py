import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ✅ Load ONLY from your saved folder
model_path = "models/bert_model"

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

model.eval()

# Input
text = input("Enter news text: ")

# Tokenize
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1)

if prediction.item() == 0:
    print("🟥 Fake News")
else:
    print("🟩 Real News")