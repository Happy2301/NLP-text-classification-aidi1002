

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

df = pd.read_csv("train.csv")
df = df[['text', 'target']].dropna()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], df['target'], test_size=0.2, random_state=42
)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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

train_labels = torch.tensor(train_labels.values)
val_labels = torch.tensor(val_labels.values)

train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

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


print("\n BERT Classification Report:")
print(classification_report(y_true, y_pred))
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
