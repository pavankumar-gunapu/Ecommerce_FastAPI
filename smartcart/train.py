"""Train and serialize purchase and customer-segmentation models."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from smartcart.data import DATA_DIR, ROOT, generate_datasets

ARTIFACT_DIR = ROOT / "artifacts"
FEATURES = ["customer_age", "total_purchases", "total_spending", "average_order_value", "website_visits", "product_price", "product_rating", "product_views", "cart_additions", "wishlist_additions", "previous_purchases", "time_spent"]
SEGMENT_FEATURES = ["total_purchases", "total_spending", "average_order_value", "website_visits", "purchase_frequency"]


def train_models() -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if not (DATA_DIR / "customer_activity.csv").exists():
        generate_datasets()
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    products = pd.read_csv(DATA_DIR / "products.csv")
    activity = pd.read_csv(DATA_DIR / "customer_activity.csv")
    dataset = (activity.merge(customers, on="customer_id", validate="many_to_one")
               .merge(products, on="product_id", validate="many_to_one"))
    dataset = dataset.drop_duplicates(subset=["customer_id", "product_id"])
    dataset = dataset[dataset.age.between(18, 100) & dataset.price.gt(0)]
    dataset = dataset.rename(columns={"age": "customer_age", "price": "product_price", "rating": "product_rating"})
    X = dataset[FEATURES].replace([np.inf, -np.inf], np.nan)
    y = dataset.purchased.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    algorithms = {
        "Logistic Regression": LogisticRegression(max_iter=1200, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=180, min_samples_leaf=3, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
    }
    metrics = {}
    fitted = {}
    for name, classifier in algorithms.items():
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ])
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics[name] = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        }
        fitted[name] = pipeline
    selected_name = max(metrics, key=lambda name: (metrics[name]["f1"], metrics[name]["roc_auc"]))
    joblib.dump({"model": fitted[selected_name], "features": FEATURES, "selected_model": selected_name, "metrics": metrics}, ARTIFACT_DIR / "purchase_model.pkl")

    customers["registration_date"] = pd.to_datetime(customers.registration_date, errors="coerce")
    age_days = (pd.Timestamp.now().normalize() - customers.registration_date).dt.days.clip(lower=1)
    segmentation = customers[SEGMENT_FEATURES[:4]].copy()
    segmentation["purchase_frequency"] = customers.total_purchases / age_days * 365
    segmentation = segmentation.replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(segmentation)
    scores = {}
    for clusters in range(2, 7):
        labels = KMeans(n_clusters=clusters, random_state=42, n_init=15).fit_predict(scaled)
        scores[clusters] = float(silhouette_score(scaled, labels))
    best_k = max(scores, key=scores.get)
    clusterer = KMeans(n_clusters=best_k, random_state=42, n_init=15).fit(scaled)
    centers = pd.DataFrame(clusterer.cluster_centers_, columns=SEGMENT_FEATURES)
    value_order = centers["total_spending"].rank(method="first").astype(int)
    segment_names = ["At-Risk Customer", "Occasional Customer", "Regular Customer", "Premium Customer"]
    names = {cluster: segment_names[min(int(value_order.iloc[cluster]) - 1, len(segment_names) - 1)] for cluster in range(best_k)}
    joblib.dump({"scaler": scaler, "model": clusterer, "features": SEGMENT_FEATURES, "segment_names": names, "best_k": best_k, "silhouette_scores": scores}, ARTIFACT_DIR / "customer_segments.pkl")
    (ARTIFACT_DIR / "model_metrics.json").write_text(json.dumps({"selected_model": selected_name, "metrics": metrics, "best_k": best_k, "silhouette_scores": scores}, indent=2))
    return {"selected_model": selected_name, "metrics": metrics, "best_k": best_k, "silhouette_scores": scores}


if __name__ == "__main__":
    results = train_models()
    print(json.dumps(results, indent=2))