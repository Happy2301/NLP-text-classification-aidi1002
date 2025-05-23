# NLP Text Classification with DistilBERT

This project is part of the AIDI 1002 Final Project. It involves building a text classification model using both traditional and transformer-based approaches.

Authors: Harpreet Singh and Ashish Acharya

## Objective

The main goal of this project is to reproduce the results of the [DistilBERT paper](https://arxiv.org/abs/1910.01108) and contribute to the paper by testing the
model on other types of datasets like IMDb review dataset which contains longer texts than the SST-2 dataset for sentiment analysis. We also do comparison of this model with older models like Word2Vec and TF-IDF.

## Project Structure

- `reproduction.ipynb` – Code for repriducing the paper on the SST-2 dataset using the trained DistilBERT model.
- `contribution.ipynb` – Applying DistilBERT to IMDb dataset to improve accuracy.
- `comparison.ipynb` - Compring transformer based approach to statistical appraches like TF-IDF and neural network based Word2Vec.
- `README.md` – Project overview and details.

## Models Used

- **DistilBERT (transformer-based)** – added as a contribution to enhance the model.
- Logistic Regression (TF-IDF)
- Word2Vec

## Dependencies

- Python 3.10+
- `torch`
- `transformers`
- `scikit-learn`
- `torch`
- `pandas`
- `numpy`

## Contribution

In this project, we reproduced and fine-tuned the DistilBERT model for text classification using the SST-2 which is used in GLUE benchmark to check accuracy of NLP systems.

Our main contribution was to evaluate the performance of the pre-trained DistilBERT model on a new dataset not originally used in the source paper. This allowed us to assess the model's generalizability and robustness in a different context with longer inputs.

Additionally, we:

Fine-tuned the model using adjusted parameters (batch size, epochs, and learning rate).

Analyzed classification metrics such as accuracy.

Compared the model’s performance before and after fine-tuning and with other models such as TF-IDF and Word2Vec for NLP tasks.

The results showed that DistilBERT maintained strong performance even on a different dataset, confirming its effectiveness as a lightweight transformer model for text classification tasks.
