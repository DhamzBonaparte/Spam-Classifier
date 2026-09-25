import streamlit as st
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import sklearn
from huggingface_hub import hf_hub_download
import os

files_to_download = ["vectorizer.pkl", "model.pkl"]

for filename in files_to_download:
    if not os.path.exists(filename):
        print(f"Downloading {filename} from Hugging Face...")
        hf_hub_download(
            repo_id="Dhamz10/Spam_Classification",
            filename=filename,
            local_dir=".",
            repo_type="dataset",
            local_dir_use_symlinks=False,
        )


nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.title("Email/SMS spam classifier")
msg = st.text_input("Enter the message...")

if st.button("Predict"):
    transformed_msg = transform_text(msg)
    vector_input = tfidf.transform([transformed_msg])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
