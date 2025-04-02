# NLP Text Classification with BERT

This project is part of the AIDI 1002 Final Project. It involves building a text classification model using both traditional and transformer-based approaches to classify disaster-related tweets.

# Objective
The main goal of this project is to classify whether a given tweet/text indicates a disaster or not.

# Project Structure
- `bert_text_classifier.py` — BERT-based text classification model.
- `bert_predict_submission.py` — Code for generating predictions on the test set using the trained BERT model.
- `bert_model.pth` — Trained BERT model weights (stored locally).
- `train.csv`, `test.csv` — Dataset files used for training and testing.
- `submission.csv` — Output predictions for submission.
- `NLP text classification model Github.ipynb` — Notebook showing traditional model pipeline (TF-IDF, Word2Vec).
- `README.md` — Project overview and details.

# Models Used
- **Logistic Regression (TF-IDF)**
- **Word2Vec**
- **BERT (transformer-based)** — added as a contribution to enhance the model.

# Results
**BERT Classification Report:**
```
Accuracy: 83.59%
Precision (class 1): 0.88
Recall (class 1): 0.71
F1-score (class 1): 0.79
```

# How to Run

1. Clone the repository:
git clone https://github.com/Happy2301/NLP-text-classification-aidi1002.git
cd NLP-text-classification-aidi1002
```

2. Install the dependencies:
pip install torch transformers scikit-learn pandas numpy
```

3. Train the BERT model:
python bert_text_classifier.py
```

4. Generate predictions and create submission file:
python bert_predict_submission.py
```

# Dependencies
- Python 3.10+
- `torch`
- `transformers`
- `scikit-learn`
- `pandas`
- `numpy`

# Author
- Harpreet Singh and Ashish Acharya
