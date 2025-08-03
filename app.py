# app.py

import streamlit as st
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nltk
from nltk.tokenize import word_tokenize

# Load vocab and model setup
with open("data/words.json", "r") as f:
    words = json.load(f)
word2idx = {word: i+1 for i, word in enumerate(words)}  # pad token = 0

with open("data/labels.json", "r") as f:
    label_map = json.load(f)  # optional: label index to category name

nltk.download('punkt')

# Define model
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

# Load model
vocab_size = len(word2idx) + 1
embed_dim = 64
num_classes = len(label_map)
model = TicketClassifier(vocab_size, embed_dim, num_classes)
model.load_state_dict(torch.load("model/ticket_classifier.pth", map_location=torch.device('cpu')))
model.eval()

# Padding function
def pad_input(sentence, seq_len=50):
    features = np.zeros((seq_len,), dtype=int)
    features[-len(sentence):] = np.array(sentence[:seq_len])
    return features

# Preprocessing
def preprocess(text):
    tokens = word_tokenize(text.lower())
    token_ids = [word2idx.get(token, 0) for token in tokens]
    return torch.tensor([pad_input(token_ids)], dtype=torch.long)

# Streamlit interface
st.title("🧾 Service Desk Ticket Classifier")
st.write("Automatically categorize service desk tickets using deep learning.")

user_input = st.text_area("Enter a customer complaint / ticket text:", height=150)

if st.button("Classify Ticket"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_tensor = preprocess(user_input)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            category = label_map[str(pred)]
            st.success(f"**Predicted Category:** {category}")