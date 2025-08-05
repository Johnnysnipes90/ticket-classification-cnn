# app.py

import streamlit as st
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nltk

import plotly.express as px

# --- Ensure 'punkt' tokenizer is available ---
import nltk
nltk.data.path.append("./nltk_data")  # Add your custom download path (if used)

# Try downloading automatically if missing
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir="./nltk_data")

from nltk.tokenize import word_tokenize

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Ticket Classifier", layout="centered", page_icon="🧾")

# --- STYLING ---
custom_css = """
<style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: None;
        border-radius: 5px;
        padding: 0.5em 1.5em;
        font-size: 16px;
    }
    .stTextArea textarea {
        border-radius: 10px;
        font-size: 16px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧾 Service Desk Ticket Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Instantly categorize support tickets using a powerful AI model</p>", unsafe_allow_html=True)
st.divider()

# --- LOAD RESOURCES ---
with open("data/words.json", "r") as f:
    words = json.load(f)
word2idx = {word: i+1 for i, word in enumerate(words)}  # Padding = 0

with open("data/labels.json", "r") as f:
    label_map = json.load(f)

# --- MODEL CLASS ---
class TicketClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TicketClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = F.relu(self.conv(x))
        x = x.mean(dim=2)
        return self.fc(x)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model = TicketClassifier(len(word2idx) + 1, 64, len(label_map))
    model.load_state_dict(torch.load("model/ticket_classifier.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# --- UTILITIES ---
def pad_input(tokens, seq_len=50):
    features = np.zeros((seq_len,), dtype=int)
    features[-len(tokens):] = np.array(tokens[:seq_len])
    return features

def preprocess(text):
    tokens = word_tokenize(text.lower())
    token_ids = [word2idx.get(token, 0) for token in tokens]
    return torch.tensor([pad_input(token_ids)], dtype=torch.long)

# --- MAIN INTERFACE ---
st.markdown("### 💬 Enter Ticket Text")
user_input = st.text_area("What did the customer say?", height=150, placeholder="e.g., I can't log into my account and need help resetting my password...")

col1, col2 = st.columns([1, 5])
with col1:
    if st.button("🔍 Classify"):
        if not user_input.strip():
            st.warning("🚫 Please enter a valid customer complaint.")
        else:
            input_tensor = preprocess(user_input)
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1).squeeze()
                pred_index = torch.argmax(probs).item()
                pred_label = label_map[str(pred_index)]

            st.success(f"🎯 **Predicted Category:** `{pred_label}`")

            st.markdown("#### 📊 Confidence Scores")
            conf_data = {
                "Category": [label_map[str(i)] for i in range(len(label_map))],
                "Confidence": [float(p) for p in probs]
            }
            fig = px.bar(conf_data, x="Category", y="Confidence", color="Category",
                         title="Model Prediction Confidence", labels={"Confidence": "Probability"},
                         height=400, width=600)
            st.plotly_chart(fig, use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.9em;'>Built with ❤️ by John Olalemi using PyTorch & Streamlit</p>", unsafe_allow_html=True)
