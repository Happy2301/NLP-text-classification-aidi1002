# NLP Text Classification with BERT

This project is part of the AIDI 1002 Final Project. It involves building a text classification model using both traditional and transformer-based approaches.

## Objective

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

## Cotribution

In this project, we reproduced and fine-tuned the DistilBERT model for text classification using the AG News dataset, which consists of over 100,000 news articles categorized into four classes: World, Sports, Business, and Sci/Tech.

Our main contribution was to evaluate the performance of the pre-trained DistilBERT model on a new dataset not originally used in the source paper. This allowed us to assess the model's generalizability and robustness in a different context.

Additionally, we:

Fine-tuned the model using adjusted parameters (batch size, epochs, and learning rate).

Analyzed classification metrics such as accuracy, precision, recall, and confusion matrix.

Compared the model’s performance before and after fine-tuning.

The results showed that DistilBERT maintained strong performance even on a different dataset, confirming its effectiveness as a lightweight transformer model for text classification tasks.

