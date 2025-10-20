# step6_prediction.py
import numpy as np

def predict_category(text, model, vectorizer, label_encoder):
    """
    Predicts category for input text using trained model.
    Returns (predicted_label, probability)
    """
    if not isinstance(text, str) or text.strip() == "":
        return ("Invalid Input", 0.0)
    
    X_vec = vectorizer.transform([text])
    probs = model.predict_proba(X_vec)[0]
    idx = np.argmax(probs)
    label = label_encoder.inverse_transform([idx])[0]
    return label, probs[idx]

text = "I was charged twice for my credit card payment"
label, prob = predict_category(text, models["LogisticRegression"], tfidf, le)
print(f"Predicted Category: {label} (Confidence: {prob:.2f})")

