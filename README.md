# 🛡️ Email / SMS Spam Classifier

An end-to-end Machine Learning web application that predicts whether a given text message or email is **Spam** or **Not Spam**. Built with Python, Scikit-Learn, Natural Language Processing (NLTK), and hosted using Streamlit.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Machine_Learning-orange)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-Hub-yellow)

---

## 🚀 Features

* **Text Preprocessing Pipeline:** Cleans input text by lowercasing, tokenizing, removing stopwords and punctuation, and applying **Porter Stemmer** normalization.
* **TF-IDF Vectorization:** Transforms raw text into meaningful numerical frequency features using a pre-trained vectorizer.
* **Cloud Model Hosting:** Automatically downloads model artifacts (`vectorizer.pkl` and `model.pkl`) securely from the **Hugging Face Hub** at runtime, keeping repository file sizes small.
* **Interactive Web Interface:** Real-time classification powered by Streamlit.

---

## 🛠️ How It Works (The Code Pipeline)

When a user enters a message and clicks **Predict**, the app triggers a 3-step pipeline:

1. **Text Transformation (`transform_text`):** Strips noise, splits words, filters out common stopwords, and stems words down to their root forms.
2. **Vectorization (`tfidf.transform`):** Converts the processed text string into a sparse numerical feature vector matching the training vocabulary.
3. **Prediction (`model.predict`):** Runs the vector through the trained machine learning model to output a binary result (`1` for Spam, `0` for Not Spam).
