📰 Fake and Real News Detection

A machine learning project that classifies news articles as Fake or Real using Natural Language Processing (NLP) and supervised machine learning techniques.

📌 Project Overview

Fake news can spread rapidly through social media and online platforms, making it difficult for readers to distinguish between reliable and misleading information.

This project uses Natural Language Processing (NLP) and machine learning to analyze the textual content of news articles and predict whether a given article is Fake or Real.

The project includes data preprocessing, text cleaning, feature extraction, model training, and prediction.

Note: This project is an educational machine-learning classifier. Its predictions should not be treated as definitive proof that a news article is true or false.

✨ Features
📰 Classifies news articles as Fake or Real
🧹 Text preprocessing and cleaning
🔤 Natural Language Processing
📊 TF-IDF-based text feature extraction
🤖 Machine learning classification
📈 Model evaluation
🖥️ Simple user interface for entering news
⚡ Fast prediction after model training
🛠️ Technologies Used
Python
Pandas — Data processing
NumPy — Numerical operations
Scikit-learn — Machine learning
NLTK — Natural Language Processing
TF-IDF — Text feature extraction

📂 Project Structure
Fake-News-Detection/
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── model/
│   └── model.pkl
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore

The exact structure may vary depending on the final implementation.

📊 Dataset

The project uses two datasets:

Fake.csv — contains fake news articles
True.csv — contains real news articles

Both datasets contain information such as:

Title
Text
Subject
Date

The datasets are combined and labeled before being used for training.

