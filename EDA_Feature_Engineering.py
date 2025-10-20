import os
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re

# ========== USER CONFIG ==========
DATA_PATH = r"C:\Users\koush\Downloads\LearningRESTAPIs\code\complaints.csv"
TEXT_COL = "Consumer complaint narrative"
LABEL_COL = "Product"
CHUNK_SIZE = 100_000
SAMPLE_ROWS = 50_000
TOP_N_WORDS = 50
# ==================================

def load_sample(path, text_col=TEXT_COL, label_col=LABEL_COL, n=SAMPLE_ROWS):
    rng = np.random.default_rng(seed=42)
    samples = []
    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
        if text_col not in chunk.columns:
            raise ValueError(f"Text column '{text_col}' not in dataset columns: {chunk.columns.tolist()}")
        k = min(len(chunk), max(1, int(n * len(chunk) / 1_000_000)))  
        samples.append(chunk.sample(n=min(k, len(chunk)), random_state=42))
        if sum(len(s) for s in samples) >= n:
            break
    sample_df = pd.concat(samples).sample(n=min(n, sum(len(s) for s in samples)), random_state=42).reset_index(drop=True)
    return sample_df

def basic_column_info(path):
    for chunk in pd.read_csv(path, chunksize=10_000):
        print("Columns:", chunk.columns.tolist())
        print("\ndtypes:\n", chunk.dtypes)
        print("\nFirst 5 rows:\n", chunk.head())
        break

def count_missing_by_col(path):
    miss = None
    total = 0
    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
        if miss is None:
            miss = chunk.isnull().sum()
        else:
            miss += chunk.isnull().sum()
        total += len(chunk)
    miss = miss.astype(int)
    miss_df = pd.DataFrame({'missing_count': miss, 'missing_pct': miss / total * 100})
    return miss_df.sort_values('missing_count', ascending=False)

def simple_tokenize(text):
    return re.findall(r'\w+', str(text).lower())

def chunk_word_counts(path, text_col=TEXT_COL, n_top=TOP_N_WORDS):
    total = Counter()
    lengths = []
    for chunk in tqdm(pd.read_csv(path, usecols=[text_col], chunksize=CHUNK_SIZE), desc="Counting words"):
        texts = chunk[text_col].astype(str)
        for t in texts:
            toks = simple_tokenize(t)
            lengths.append(len(toks))
            total.update(toks)
    most_common = total.most_common(n_top)
    return most_common, lengths

def compute_basic_text_features(df, text_col=TEXT_COL):
    s = df[text_col].astype(str)
    df_features = pd.DataFrame({
        'char_len': s.str.len(),
        'word_len': s.apply(lambda x: len(simple_tokenize(x))),
        'punct_count': s.apply(lambda x: len(re.findall(r'[^\w\s]', x))),
        'upper_ratio': s.apply(lambda x: sum(1 for ch in x if ch.isupper()) / (len(x) + 1e-9)),
    })
    return df_features

if __name__ == "__main__":
    print("=== Basic column info (first chunk) ===")
    basic_column_info(DATA_PATH)

    print("\n=== Loading sample for quick EDA ===")
    sample = load_sample(DATA_PATH, TEXT_COL, LABEL_COL, n=SAMPLE_ROWS)
    print(sample.head())

    print("\n=== Missing values (chunked) ===")
    miss_df = count_missing_by_col(DATA_PATH)
    print(miss_df.head(20))

    if LABEL_COL in sample.columns:
        print("\n=== Label distribution (sample) ===")
        print(sample[LABEL_COL].value_counts(dropna=False))
        label_counts = Counter()
        for chunk in pd.read_csv(DATA_PATH, usecols=[LABEL_COL], chunksize=CHUNK_SIZE):
            label_counts.update(chunk[LABEL_COL].astype(str).fillna("<<NA>>"))
        print("\n=== Label distribution (full, chunked) ===")
        for label, cnt in label_counts.most_common(20):
            print(f"{label}: {cnt}")

    print("\n=== Computing top words and text-length distribution (chunked) ===")
    top_words, lengths = chunk_word_counts(DATA_PATH, TEXT_COL, n_top=TOP_N_WORDS)
    print("\nTop words (top 30):", top_words[:30])

    try:
        feats = compute_basic_text_features(sample, TEXT_COL)
        fig, axes = plt.subplots(2,2, figsize=(12,8))
        sns.histplot(feats['word_len'], bins=50, ax=axes[0,0]).set_title('Word count (sample)')
        sns.histplot(feats['char_len'], bins=50, ax=axes[0,1]).set_title('Char length (sample)')
        sns.histplot(feats['punct_count'], bins=30, ax=axes[1,0]).set_title('Punctuation count (sample)')
        sns.boxplot(x=feats['word_len'], ax=axes[1,1]).set_title('Word count boxplot (sample)')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Plotting failed:", e)

    top_df = pd.DataFrame(top_words, columns=['word','count'])
    top_df.to_csv("top_words.csv", index=False)
    print("\nSaved top words to top_words.csv")

    sample_feats = compute_basic_text_features(sample, TEXT_COL)
    sample_out = pd.concat([sample.reset_index(drop=True), sample_feats], axis=1)
    sample_out.to_csv("sample_with_text_features.csv", index=False)
    print("Saved sample_with_text_features.csv")

