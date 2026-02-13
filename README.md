# End-to-End MLOps ML Pipeline

This project implements a complete, automated machine learning pipeline for **Email Classification**, utilizing MLOps best practices with **DVC (Data Version Control)**. It covers every stage of the lifecycle, from raw data ingestion to model evaluation, ensuring reproducibility and versioning of data and models.

## 🚀 Project Overview

The pipeline is designed to classify emails (e.g., Spam vs. Ham) using a structured workflow. By using DVC, each stage is version-controlled, allowing you to track experiments and easily reproduce results.

### 🏗️ Directory Structure

```text
├── data
│   ├── raw/          # Initial data split (Train/Test)
│   ├── interim/      # Text-preprocessed data
│   └── processed/    # Vectorized (TF-IDF) feature sets
├── experiments/      # Source datasets and Jupyter Notebooks
├── logs/             # Component-level execution logs
├── models/           # Trained models and evaluation metrics
├── src/              # Source code for pipeline stages
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── feature_engineerning.py
│   ├── model_training.py
│   └── model_evluation.py
├── dvc.yaml          # DVC pipeline configuration
└── params.yaml        # Pipeline hyperparameters
```

## 🛠️ Pipeline Stages

The pipeline consists of 5 main stages, as defined in `dvc.yaml`:

1.  **Data Ingestion**: Loads the raw dataset from `experiments/email.csv` and splits it into training and testing sets in `data/raw/`.
2.  **Data Preprocessing**: Cleans the text data (lowercasing, tokenization, stop-word removal, and stemming) using NLTK, saving results to `data/interim/`.
3.  **Feature Engineering**: Converts cleaned text into numerical features using **TF-IDF Vectorization**, stored in `data/processed/`.
4.  **Model Training**: Trains a Linear **SVM (Support Vector Machine)** model on the vectorized features and saves the model to `models/model.pkl`.
5.  **Model Evaluation**: Evaluates the model on the test set, generating metrics (Accuracy, Classification Report, Confusion Matrix) in `models/metrics.json`.

## ⚙️ Setup & Usage

### 📦 Prerequisites

Ensure you have Python installed, then install the necessary dependencies:

```bash
pip install pandas numpy scikit-learn nltk dvc
```

### 🏃 Running the Pipeline

To run the entire pipeline end-to-end, simply use DVC:

```bash
dvc repro
```

This will automatically check which stages need re-running based on changes in code or data.

### 📊 Viewing Reproducibility & Metrics

You can view the tracked metrics after a run:

```bash
dvc metrics show
```

## 🧠 Technologies Used

- **Python**: Core programming language.
- **Scikit-learn**: Machine learning model and preprocessing.
- **NLTK**: Natural Language Toolkit for advanced text processing.
- **DVC**: Version control for data and ML pipelines.
- **Pandas/NumPy**: Data manipulation and numerical operations.

---
*Developed as part of an End-to-End MLOps Pipeline project.*
