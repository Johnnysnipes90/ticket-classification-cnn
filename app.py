# app.py

import streamlit as st
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from PIL import Image

# --- INITIAL SETUP ---
st.set_page_config(page_title="Ticket Classifier", layout="centered", page_icon="🧾")

# Optional custom header image
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧾 Service Desk Ticket Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict the category of customer complaints using a deep learning model (CNN)</p>", unsafe_allow_html=True)
st.divider()

# Load vocab
with open("data/words.json", "r") as f:
    words = json.load(f)
word2idx = {word: i+1 for i, word in enumerate(words)}  # Reserve 0 for padding

# Load label map
with open("data/labels.json", "r") as f:
    label_map = json.load(f)

nltk.download('punkt', quiet=True)

# --- MODEL DEFINITION ---
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

# Load trained model
vocab_size = len(word2idx) + 1
embed_dim = 64
num_classes = len(label_map)

model = TicketClassifier(vocab_size, embed_dim, num_classes)
model.load_state_dict(torch.load("model/ticket_classifier.pth", map_location="cpu"))
model.eval()

# --- HELPER FUNCTIONS ---
def pad_input(tokens, seq_len=50):
    features = np.zeros((seq_len,), dtype=int)
    features[-len(tokens):] = np.array(tokens[:seq_len])
    return features

def preprocess(text):
    tokens = word_tokenize(text.lower())
    token_ids = [word2idx.get(token, 0) for token in tokens]
    return torch.tensor([pad_input(token_ids)], dtype=torch.long)

# --- MAIN INTERFACE ---
st.markdown("### ✍️ Enter Ticket Text")
user_input = st.text_area("Type or paste a customer complaint:", height=150, placeholder="e.g., I was charged twice on my credit card...")

col1, col2 = st.columns([1, 3])
with col1:
    submit = st.button("🔍 Classify")

if submit:
    if not user_input.strip():
        st.warning("⚠️ Please enter a complaint or ticket text.")
    else:
        input_tensor = preprocess(user_input)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1).squeeze()
            pred_index = torch.argmax(probs).item()
            pred_label = label_map[str(pred_index)]

        st.success(f"🎯 **Predicted Category:** `{pred_label}`")
        st.markdown("#### 🔢 Prediction Confidence")
        confidence_df = {
            label_map[str(i)]: float(probs[i].item()) for i in range(len(label_map))
        }
        st.bar_chart(confidence_df)

        st.markdown("---")
        with st.expander("📄 Model Summary"):
            st.text(model)

        with st.expander("ℹ️ About this App"):
            st.markdown("""
            - Built with PyTorch and Streamlit
            - Uses a 1D Convolutional Neural Network
            - Trained on tokenized customer complaint texts
            - Pads/truncates inputs to length 50
            """)
