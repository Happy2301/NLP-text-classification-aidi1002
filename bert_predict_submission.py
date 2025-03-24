import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load test data
df_test = pd.read_csv("test.csv")
df_test['clean_text'] = df_test['text'].fillna("").apply(lambda x: str(x).lower())

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.load_state_dict(torch.load("bert_model.pth", map_location=torch.device('cpu')))
model.eval()

# Tokenize test data
tokens = tokenizer(list(df_test['clean_text']), truncation=True, padding=True, return_tensors='pt', max_length=128)

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
tokens = {k: v.to(device) for k, v in tokens.items()}

# Run prediction
with torch.no_grad():
    outputs = model(**tokens)
    predictions = torch.argmax(outputs.logits, axis=1).cpu().numpy()

# Save submission
df_test['target'] = predictions
submission = df_test[['id', 'target']]
submission.to_csv("submission.csv", index=False)
print("submission.csv generated!")

