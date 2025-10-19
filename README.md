# 🧠 Text Classification on Consumer Complaint Dataset

### 📘 Project Overview
This project focuses on building a **multi-class text classification model** that categorizes consumer complaints into specific categories such as:
- **Credit reporting, repair, or other**
- **Debt collection**
- **Consumer Loan**
- **Mortgage**

The complete pipeline follows a structured Machine Learning workflow:
1. **Exploratory Data Analysis (EDA) & Feature Engineering**
2. **Text Pre-Processing**
3. **Model Selection**
4. **Model Performance Comparison**
5. **Model Evaluation**
6. **Prediction**

---

## 📊 1. Exploratory Data Analysis (EDA) & Feature Engineering

The first step focuses on understanding the dataset and engineering useful text-based features.

**Key tasks:**
- Load the dataset in chunks (1.5 GB — memory optimized).
- Inspect column structure and missing values.
- Analyze label distribution.
- Compute word frequencies and text lengths.
- Feature engineering:
  - Text length (in words and characters)
  - Punctuation count
  - Uppercase ratio

**Generated Outputs:**
- `top_words.csv` — top frequent words across the dataset  
- `sample_with_text_features.csv` — small sample file for model prototyping

📁 **Code file:** `step1_eda.py`

---

## 🧹 2. Text Preprocessing

Clean and normalize raw text for model training.

**Steps included:**
- Lowercasing text  
- Removing punctuation, digits, URLs, and extra spaces  
- Removing stopwords  
- Lemmatization using spaCy or stemming using NLTK  
- Tokenization for model input  

**Optional Enhancements:**
- Handling emojis and special symbols  
- Removing rare or overly common words  

📁 **Code file:** `step2_preprocessing.py`

---

## 🤖 3. Model Selection (Multi-Class Classification)

We train and compare multiple models to identify the best-performing one.

**Models used:**
- Logistic Regression (TF-IDF features)  
- Multinomial Naive Bayes  
- Random Forest  
- XGBoost  
- Transformer-based models (DistilBERT / BERT for fine-tuning)

📁 **Code file:** `step3_model_selection.py`

---

## ⚖️ 4. Model Performance Comparison

Compare models on multiple metrics to identify trade-offs.

**Metrics:**
- Accuracy  
- Precision, Recall, F1-score  
- Confusion Matrix  
- Execution time  

**Visualizations:**
- Confusion Matrix heatmap  
- F1-score bar chart  

📁 **Code file:** `step4_model_comparison.py`

---

## 📈 5. Model Evaluation

Evaluate the final model on test data and analyze performance.

**Tasks:**
- Generate classification report  
- Visualize confusion matrix and ROC/AUC  
- Identify misclassified examples  
- Analyze per-class accuracy and errors  

📁 **Code file:** `step5_evaluation.py`

---

## 🔮 6. Prediction

Implement a simple interface for predicting categories on new unseen text.

**Example function:**
```python
def predict_category(text):
    """
    Predicts the category of a given complaint text.
    Returns (category_name, probability).
    """
