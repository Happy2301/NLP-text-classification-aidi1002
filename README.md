# NLP Text Classification with BERT

This project is part of the AIDI 1002 Final Project. It involves building a text classification model using both traditional and transformer-based approaches.

Authors: Harpreet Sandhu and Ashish Acharya

## Objective

The main goal of this project is to classify whether a given tweet/text indicates a disaster or not.

## Project Structure

- `reproduction.ipynb` – Code for repriducing the paper on the SST-2 dataset using the trained DistilBERT model.
- `contribution.ipynb` – Applying DistilBERT to IMDb dataset to improve accuracy.
- `comparison.ipynb` - Compring transformer based approach to statistical appraches like word2vec and TF-IDF.
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

Our main contribution was to evaluate the performance of the pre-trained DistilBERT model on a new dataset not originally used in the source paper. This allowed us to assess the model's generalizability and robustness in a different context.

Additionally, we:

Fine-tuned the model using adjusted parameters (batch size, epochs, and learning rate).

Analyzed classification metrics such as accuracy.

Compared the model’s performance before and after fine-tuning and with other models such as TF-IDF and Word2Vec for NLP tasks.

The results showed that DistilBERT maintained strong performance even on a different dataset, confirming its effectiveness as a lightweight transformer model for text classification tasks.
