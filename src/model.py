import os
import json
import numpy as np
import pandas as pd
from src.data_validator import generate_statistics, validate_statistics, compute_drift_and_skew

class CreditRiskModel:
    def __init__(self, lr=0.08, n_epochs=300):
        self.lr = lr
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = 0.0
        self.num_means = {}
        self.num_stds = {}
        self.cat_categories = {}

    def _prep(self, df, is_train=False):
        parts = []
        num_cols = ['customer_age', 'annual_income', 'credit_score', 'loan_amount', 'employment_length_years', 'debt_to_income_ratio', 'account_tenure_months', 'risk_score_external']
        cat_cols = ['loan_intent', 'home_ownership', 'historical_default']
        for col in num_cols:
            if col in df.columns:
                s = df[col].values.astype(float)
                if is_train:
                    self.num_means[col] = float(np.nanmean(s))
                    self.num_stds[col] = float(np.nanstd(s) + 1e-7)
                mean_v = self.num_means.get(col, 0.0)
                std_v = self.num_stds.get(col, 1.0)
                s = np.where(np.isnan(s), mean_v, s)
                parts.append(((s - mean_v) / std_v).reshape(-1, 1))
        for col in cat_cols:
            if col in df.columns:
                s = df[col].fillna("UNKNOWN").astype(str)
                if is_train:
                    self.cat_categories[col] = sorted(list(s.unique()))
                for cat in self.cat_categories.get(col, []):
                    parts.append(((s == cat).astype(float).values).reshape(-1, 1))
        return np.hstack(parts)

    def fit(self, df, target='is_default'):
        y = df[target].values.astype(float)
        X = self._prep(df, True)
        self.weights = np.zeros(X.shape[1])
        for _ in range(self.n_epochs):
            p = 1.0 / (1.0 + np.exp(-np.clip(np.dot(X, self.weights) + self.bias, -12, 12)))
            err = p - y
            self.weights -= self.lr * ((np.dot(X.T, err) / len(y)) + 0.01 * self.weights)
            self.bias -= self.lr * np.mean(err)

    def predict_proba(self, df):
        X = self._prep(df, False)
        return 1.0 / (1.0 + np.exp(-np.clip(np.dot(X, self.weights) + self.bias, -12, 12)))

    def evaluate(self, df, y_true):
        p = self.predict_proba(df)
        pred = (p >= 0.5).astype(int)
        acc = float(np.mean(pred == y_true))
        pos, neg = p[y_true == 1], p[y_true == 0]
        ranks = np.argsort(np.argsort(p))
        roc_auc = float((np.sum(ranks[y_true == 1]) - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))) if len(pos)*len(neg) else 0.5
        return {"accuracy": round(acc, 4), "roc_auc": round(roc_auc, 4)}

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"weights": self.weights.tolist(), "bias": self.bias, "num_means": self.num_means, "num_stds": self.num_stds, "cat_categories": self.cat_categories}, f)

class DataValidationGate:
    def __init__(self, schema, baseline_stats, drift_threshold=0.08):
        self.schema = schema
        self.baseline_stats = baseline_stats
        self.drift_threshold = drift_threshold

    def inspect_batch(self, df, environment="SERVING"):
        stats = generate_statistics(df)
        anomalies = validate_statistics(stats, self.schema, environment)
        drift = compute_drift_and_skew(self.baseline_stats, stats, self.schema)
        is_passed = (len(anomalies) == 0) and (len(drift["alerts"]) == 0)
        status = "PASSED" if is_passed else ("BLOCKED" if any(a["severity"] == "ERROR" for a in anomalies) else "WARNING_DRIFT")
        return {"status": status, "anomalies": anomalies, "drift_alerts": drift["alerts"]}

    def predict_guarded(self, model, df):
        gate = self.inspect_batch(df, "SERVING")
        probs = model.predict_proba(df)
        return {"gate_status": gate["status"], "mean_prob": round(float(np.mean(probs)), 4)}
