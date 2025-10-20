import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "sample_cleaned.csv"
TEXT_COL = "cleaned_text"
LABEL_COL = "Product"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_SAMPLES_PER_CLASS = 50
MAX_TFIDF_FEATURES = 10000

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows")

# Merge rare classes
class_counts = df[LABEL_COL].value_counts()
rare_classes = class_counts[class_counts < MIN_SAMPLES_PER_CLASS].index
df[LABEL_COL] = df[LABEL_COL].apply(lambda x: x if x not in rare_classes else "Other")
print(f"Remaining classes after merging rare ones: {df[LABEL_COL].nunique()}")

le = LabelEncoder()
y_enc = le.fit_transform(df[LABEL_COL])
X = df[TEXT_COL].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
)

print("\n=== TF-IDF Feature Extraction ===")
tfidf = TfidfVectorizer(max_features=MAX_TFIDF_FEATURES, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE),
    "MultinomialNB": MultinomialNB(),
    "RandomForest": RandomForestClassifier(n_estimators=200, class_weight='balanced_subsample', random_state=RANDOM_STATE)
}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"{name} Accuracy: {acc:.4f}, F1-weighted: {f1:.4f}")
    labels_in_test = sorted(list(set(y_test)))
    target_names = [le.classes_[i] for i in labels_in_test]
    print(classification_report(y_test, y_pred, labels=labels_in_test, target_names=target_names))

print("\nAll classical models done.")

