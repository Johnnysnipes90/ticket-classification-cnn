# ticket_classifier.py

import json
import numpy as np
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Accuracy, Precision, Recall

# Download punkt tokenizer (used by nltk if needed)
nltk.download('punkt')

# Load data
with open("data/words.json", "r") as f:
    words = json.load(f)
with open("data/text.json", "r") as f:
    text = json.load(f)
labels = np.load("data/labels.npy")

# Create vocabulary mappings
word2idx = {word: i+1 for i, word in enumerate(words)}  # reserve 0 for padding
idx2word = {i+1: word for i, word in enumerate(words)}

# Convert words to indices
for i, sentence in enumerate(text):
    text[i] = [word2idx.get(word, 0) for word in sentence]

# Pad sequences
def pad_input(sequences, seq_len=50):
    features = np.zeros((len(sequences), seq_len), dtype=int)
    for i, seq in enumerate(sequences):
        features[i, -len(seq):] = np.array(seq[:seq_len])
    return features

text = pad_input(text, 50)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    text, labels, test_size=0.2, random_state=42)

train_data = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).long())
test_data = TensorDataset(torch.tensor(X_test), torch.tensor(y_test).long())

train_loader = DataLoader(train_data, batch_size=400, shuffle=True)
test_loader = DataLoader(test_data, batch_size=400, shuffle=False)

# Define model
class TicketClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TicketClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        x = F.relu(self.conv(x))                # (batch, embed_dim, seq_len)
        x = x.mean(dim=2)                       # Global average pooling
        return self.fc(x)

# Model setup
vocab_size = len(word2idx) + 1
embed_dim = 64
num_classes = len(np.unique(labels))
model = TicketClassifier(vocab_size, embed_dim, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# Train model
model.train()
for epoch in range(3):
    total_loss = 0
    total_samples = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
    avg_loss = total_loss / total_samples
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")

# Evaluate model
model.eval()
predictions = []
true_labels = []

accuracy_metric = Accuracy(task='multiclass', num_classes=num_classes)
precision_metric = Precision(task='multiclass', num_classes=num_classes, average=None)
recall_metric = Recall(task='multiclass', num_classes=num_classes, average=None)

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        predictions.extend(preds.tolist())
        true_labels.extend(targets.tolist())
        accuracy_metric.update(preds, targets)
        precision_metric.update(preds, targets)
        recall_metric.update(preds, targets)

accuracy = accuracy_metric.compute().item()
precision = precision_metric.compute().tolist()
recall = recall_metric.compute().tolist()

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision (per class): {precision}")
print(f"Recall (per class): {recall}")

# Save model
torch.save(model.state_dict(), "model/ticket_classifier.pth")
print("Model saved to model/ticket_classifier.pth")