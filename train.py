"""
Machine Learning Assignment - 2
BITS Pilani - WILP | M.Tech (AIML)
Model Training and Evaluation Pipeline for UCI Spambase Dataset
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

def run_pipeline():
    print("=" * 60)
    print("1. Fetching UCI Spambase Dataset...")
    print("=" * 60)
    
    # Fetch spambase (UCI ID 94, OpenML ID 44)
    data = fetch_openml(name='spambase', version=1, as_frame=True)
    df = data.frame.copy()
    df['class'] = df['class'].astype(int)
    
    # Normalize column names to avoid special characters
    rename_map = {
        ';': 'semicolon',
        '(': 'round_bracket',
        '[': 'square_bracket',
        '!': 'exclamation',
        '$': 'dollar',
        '#': 'hash'
    }
    new_cols = []
    for col in df.columns:
        c = col
        for k, v in rename_map.items():
            c = c.replace(k, v)
        new_cols.append(c)
    df.columns = new_cols
    
    print(f"Dataset shape: {df.shape[0]} instances, {df.shape[1] - 1} features + 1 target.")
    print(f"Class distribution: {dict(df['class'].value_counts())}")
    
    os.makedirs('model', exist_ok=True)
    
    X = df.drop(columns=['class'])
    y = df['class']
    
    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
    
    # Save test_data.csv (features + target)
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv('test_data.csv', index=False)
    print("Saved 'test_data.csv' successfully.")
    
    # Standardize features for distance and gradient-sensitive models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'model/scaler.joblib')
    print("Saved 'model/scaler.joblib' successfully.")
    
    # Define models
    models = {
        'Logistic Regression': (
            LogisticRegression(max_iter=1000, random_state=42),
            True,
            'logistic_regression.joblib'
        ),
        'Decision Tree': (
            DecisionTreeClassifier(max_depth=6, random_state=42),
            False,
            'decision_tree.joblib'
        ),
        'KNN': (
            KNeighborsClassifier(n_neighbors=5),
            True,
            'knn.joblib'
        ),
        'Naive Bayes': (
            GaussianNB(),
            False,
            'naive_bayes.joblib'
        ),
        'Support Vector Machine': (
            SVC(kernel='rbf', probability=True, random_state=42),
            True,
            'svm.joblib'
        )
    }
    
    results = []
    
    print("\n" + "=" * 60)
    print("2. Training and Evaluating Models...")
    print("=" * 60)
    
    for name, (clf, use_scaled, filename) in models.items():
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test
        
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        joblib.dump(clf, os.path.join('model', filename))
        
        results.append({
            'ML Model Name': name,
            'Accuracy': round(acc, 4),
            'AUC': round(auc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4),
            'MCC': round(mcc, 4)
        })
        
        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.4f} | AUC: {auc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("3. Final Comparison Table:")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    results_df.to_csv('evaluation_results.csv', index=False)
    print("\nResults exported to 'evaluation_results.csv'.")

if __name__ == '__main__':
    run_pipeline()
