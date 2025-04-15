# 🧠 Real vs. Fake Job Detection using NLP

This project uses Natural Language Processing (NLP) and machine learning to detect fraudulent job postings. It leverages TF-IDF vectorization, a Random Forest classifier, and a comprehensive NLP pipeline to classify job descriptions as real or fake.

---

## 📌 Project Overview

Fake job ads pose serious threats to job seekers. This project builds a binary classification model using real-world job postings data to identify scams. It includes preprocessing, feature engineering, model training, evaluation, and deployment plans.

---

## 📁 Dataset

The dataset includes job postings with the following features:
- `title`, `location`, `department`, `salary_range`
- `company_profile`, `description`, `requirements`, `benefits`
- `telecommuting`, `has_company_logo`, `fraudulent` (target)

Only a small portion of the data is fraudulent, making class imbalance a challenge.

---

## 🔧 NLP Pipeline

1. **Text Cleaning** – Lowercasing, removing punctuation/stopwords, lemmatization  
2. **TF-IDF Vectorization** – Applied on major text fields  
3. **Train/Test Split** – 80/20  
4. **Model** – Random Forest Classifier  
5. **Evaluation Metrics** – Accuracy, Precision, Recall, F1, AUC-ROC  
6. **Visualization** – Confusion Matrix and metric plots

---

## 📊 Model Performance

| Metric       | Score      |
|--------------|------------|
| Accuracy     | 96.5%      |
| Precision    | 1.00       |
| Recall       | 0.277      |
| F1 Score     | 0.434      |
| AUC-ROC      | 0.639      |

> *Note: Class imbalance leads to low recall despite high accuracy.*

---

## 🚀 Deployment Plan

- Model saved as `random_forest_model.pkl`
- FastAPI backend with REST endpoint for predictions
- UI/UX integration for recruiters or job boards
- Periodic retraining with updated data

---

## 📦 Installation

```bash
git clone https://github.com/<your-username>/real-fake-job-nlp.git
cd real-fake-job-nlp
pip install -r requirements.txt
