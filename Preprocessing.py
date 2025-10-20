import pandas as pd
import re
from tqdm import tqdm
import spacy
from nltk.corpus import stopwords
import nltk
import os

DATA_PATH = "sample_with_text_features.csv"
TEXT_COL = "Consumer complaint narrative"
PROCESSED_PATH = "sample_cleaned.csv"
REMOVE_STOPWORDS = True
USE_LEMMATIZATION = True

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# Load spacy
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Keep some punctuation relevant for finance: #, $, +
    text = re.sub(r"[^a-z0-9\s#\$+]", " ", text)
    tokens = text.split()
    if REMOVE_STOPWORDS:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if USE_LEMMATIZATION:
        doc = nlp(" ".join(tokens))
        tokens = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
    return " ".join(tokens)

if __name__ == "__main__":
    print("=== Loading data ===")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")

    print("=== Cleaning text ===")
    tqdm.pandas(desc="Cleaning text")
    df["cleaned_text"] = df[TEXT_COL].progress_apply(clean_text)

    df = df[df["cleaned_text"].str.strip() != ""]
    print(f"Rows remaining after cleaning: {len(df)}")

    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved cleaned data to {PROCESSED_PATH}")

