import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load model
model_path = "models/bert_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

model.eval()

# UI
st.title("📰 Fake News Detection (BERT)")
st.write("Enter a news article below:")

user_input = st.text_area("News Text")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64
        )

        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1)

        if prediction.item() == 0:
            st.error("🟥 Fake News")
        else:
            st.success("🟩 Real News")