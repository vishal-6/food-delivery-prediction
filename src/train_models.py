import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import generate_dataset
from feature_engineering import engineer_features, get_features_and_target, scale_features

def prepare_data():
    df = generate_dataset()
    df = engineer_features(df)
    X, y, feature_cols = get_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    print(f"✅ Train size: {X_train_scaled.shape}, Test size: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols

def train_naive_bayes(X_train, y_train):
    print("\n🔵 Training Naive Bayes...")
    model = GaussianNB()
    model.fit(X_train, y_train)
    print("✅ Naive Bayes trained!")
    return model

def train_knn(X_train, y_train):
    print("\n🟡 Training KNN...")
    params = {"n_neighbors": [3, 5, 7, 9]}
    knn = GridSearchCV(KNeighborsClassifier(), params, cv=3, n_jobs=-1)
    knn.fit(X_train, y_train)
    print(f"✅ Best params: {knn.best_params_}")
    return knn.best_estimator_

def train_decision_tree(X_train, y_train):
    print("\n🌳 Training Decision Tree...")
    params = {
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10]
    }
    dt = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=3, n_jobs=-1)
    dt.fit(X_train, y_train)
    print(f"✅ Best params: {dt.best_params_}")
    return dt.best_estimator_

def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n📊 {name} Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred))
    return acc

def save_models(models):
    os.makedirs("outputs", exist_ok=True)
    for name, model in models.items():
        path = f"outputs/{name}.pkl"
        joblib.dump(model, path)
        print(f"✅ Saved {name} → {path}")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_data()

    nb  = train_naive_bayes(X_train, y_train)
    knn = train_knn(X_train, y_train)
    dt  = train_decision_tree(X_train, y_train)

    models = {"naive_bayes": nb, "knn": knn, "decision_tree": dt}

    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)

    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_test, y_test, name)

    best = max(results, key=results.get)
    print(f"\n🏆 Best Model: {best} with {results[best]*100:.2f}% accuracy")

    save_models(models) 