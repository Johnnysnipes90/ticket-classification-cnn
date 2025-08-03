# 🧠 Service Desk Ticket Classification with Deep Learning

Efficient handling of service desk tickets is crucial for organizations. This project uses a **Convolutional Neural Network (CNN)** to classify service desk complaints into categories like:

- Mortgage
- Credit Card
- Debt Collection
- Money Transfers
- Others
# ✅ Local Setup Instructions
1. 📁 Folder Structure
ticket-classification-cnn/
```
│
├── data/
│   ├── words.json
│   ├── text.json
│   └── labels.npy
│
├── model/
│   └── ticket_classifier.pth        
│
├── ticket_classifier.py              # Main script
├── requirements.txt
└── README.md
```

2. 🐍 requirements.txt
Create a file named requirements.txt with the following:
```
torch>=2.0.0
torchmetrics>=0.11.0
scikit-learn
numpy
pandas
nltk
```

Install it using:
```
pip install -r requirements.txt
```

## 💡 Project Highlights

- Preprocessing with custom vocabulary
- CNN-based text classification
- Evaluation with Accuracy, Precision, Recall
- Built using PyTorch + TorchMetrics

## 📊 Results

| Metric    | Score (%) |
|-----------|-----------|
| Accuracy  | 79.7      |
| Precision | per-class |
| Recall    | per-class |

## 🧾 Example Output

```txt
Epoch: 1, Loss: 0.00385
Epoch: 2, Loss: 0.00153
Epoch: 3, Loss: 0.00079
Accuracy: 0.7969
Precision (per class): [0.62, 0.76, 0.80, 0.87, 0.95]
Recall (per class):    [0.73, 0.71, 0.88, 0.79, 0.84]
