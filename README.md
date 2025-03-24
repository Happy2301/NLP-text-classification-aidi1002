# NLP Text Classification with BERT

This project is part of the AIDI 1002 Final Project. It involves building a text classification model using both traditional and transformer-based approaches.

## 🔍 Objective

The main goal of this project is to classify whether a given tweet/text indicates a disaster or not.

## Project Structure

- `bert_text_classifier.py` – BERT-based text classification model.
- `bert_predict_submission.py` – Code for generating predictions on the test set using the trained BERT model.
- `bert_model.pth` – Trained BERT model weights (stored locally).
- `train.csv`, `test.csv` – Dataset files used for training and testing.
- `submission.csv` – Output predictions for submission.
- `NLP text classification model Github.ipynb` – Notebook showing traditional model pipeline.
- `README.md` – Project overview and details.

## Models Used

- Logistic Regression (TF-IDF)
- Word2Vec
- **BERT (transformer-based)** – added as a contribution to enhance the model.

## Dependencies

- Python 3.10+
- `torch`
- `transformers`
- `scikit-learn`
- `pandas`
- `numpy`

