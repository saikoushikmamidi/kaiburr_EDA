# step4_model_comparison.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Load saved model performance results (optional)
# Or you can pass from previous script if run together
from sklearn.metrics import accuracy_score, f1_score

# Assume you have predictions and true labels from step 3
# For example:
# y_test, y_pred_dict, le = <from step 3>
# where y_pred_dict = {"LogisticRegression": y_pred_lr, "MultinomialNB": y_pred_nb, ...}

# === Example reusable function for performance plotting ===
def compare_model_performance(results_dict):
    """
    results_dict = {'ModelName': {'accuracy': 0.9, 'f1': 0.88}, ...}
    """
    df = pd.DataFrame(results_dict).T.reset_index().rename(columns={'index': 'Model'})
    
    plt.figure(figsize=(8,5))
    sns.barplot(data=df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                x='Model', y='Score', hue='Metric')
    plt.title("Model Performance Comparison")
    plt.ylim(0, 1)
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.show()
    return df

def plot_confusion_matrix(model_name, y_true, y_pred, le):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap='Blues', xticks_rotation=90)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.show()

# Example usage after step 3:
# results = {
#     "LogisticRegression": {"accuracy": acc_lr, "f1": f1_lr},
#     "MultinomialNB": {"accuracy": acc_nb, "f1": f1_nb},
#     "RandomForest": {"accuracy": acc_rf, "f1": f1_rf},
# }
# compare_model_performance(results)
# plot_confusion_matrix("LogisticRegression", y_test, y_pred_lr, le)

