# BERT Text Classifier for AIDI 1002 Final Project
# --------------------------------------------------
# This script loads train.csv, tokenizes with BERT, trains a classifier, and prints evaluation metrics

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Load dataset
df = pd.read_csv("train.csv")
df = df[['text', 'target']].dropna()

# Split the data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], df['target'], test_size=0.2, random_state=42
)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize function
def tokenize(texts):
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=128
    )

train_encodings = tokenize(train_texts)
val_encodings = tokenize(val_texts)

# Convert labels to tensors
train_labels = torch.tensor(train_labels.values)
val_labels = torch.tensor(val_labels.values)

# Create datasets
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)

# Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Training loop
model.train()
for epoch in range(1):
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        b_input_ids, b_input_mask, b_labels = [x.to(device) for x in batch]

        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())

# Evaluation
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for batch in val_loader:
        b_input_ids, b_input_mask, b_labels = [x.to(device) for x in batch]
        outputs = model(b_input_ids, attention_mask=b_input_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        y_true.extend(b_labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())

# Metrics
print("\n✅ BERT Classification Report:")
print(classification_report(y_true, y_pred))
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
