# step5_model_evaluation.py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle

def plot_multiclass_roc(model_name, y_test, y_score, le):
    y_test_bin = label_binarize(y_test, classes=range(len(le.classes_)))
    n_classes = y_test_bin.shape[1]

    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(7,6))
    colors = cycle(plt.cm.tab10.colors)
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{le.classes_[i]} (AUC = {roc_auc[i]:0.2f})')
    plt.plot([0,1], [0,1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multi-Class ROC Curve ({model_name})')
    plt.legend(loc='lower right', fontsize='small')
    plt.tight_layout()
    plt.show()

def show_misclassified_examples(X_test, y_test, y_pred, le, n=10):
    mis_idx = np.where(y_test != y_pred)[0]
    print(f"\nShowing {min(n, len(mis_idx))} misclassified examples:\n")
    for i in mis_idx[:n]:
        print(f"True: {le.classes_[y_test[i]]} | Predicted: {le.classes_[y_pred[i]]}")
        print(f"Text: {X_test[i][:300]}...")
        print("-"*80)

