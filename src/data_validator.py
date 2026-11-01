import os
import json
import numpy as np
import pandas as pd

class FeatureSchema:
    def __init__(self, name, feature_type, min_fraction=1.0, domain=None, min_val=None, max_val=None, not_in_environment=None):
        self.name = name
        self.feature_type = feature_type
        self.min_fraction = min_fraction
        self.domain = set(domain) if domain else set()
        self.min_val = min_val
        self.max_val = max_val
        self.min_domain_mass = 1.0
        self.not_in_environment = set(not_in_environment) if not_in_environment else set()
        self.drift_threshold_l_inf = None

class DatasetSchema:
    def __init__(self):
        self.features = {}
        self.default_environment = ["TRAINING", "SERVING"]

    def get_feature(self, name):
        return self.features.get(name)

    def add_feature(self, f):
        self.features[f.name] = f

    def to_pbtxt(self):
        lines = ["# Proto: tensorflow_metadata.proto.v0.Schema", ""]
        for env in self.default_environment:
            lines.append(f'default_environment: "{env}"')
        lines.append("")
        for f in self.features.values():
            lines.append("feature {")
            lines.append(f'  name: "{f.name}"')
            lines.append(f'  type: {f.feature_type}')
            if f.min_fraction < 1.0:
                lines.append(f"  presence {{ min_fraction: {f.min_fraction:.2f} }}")
            if f.domain:
                lines.append("  domain_info {")
                if f.min_domain_mass < 1.0:
                    lines.append(f"    min_domain_mass: {f.min_domain_mass:.2f}")
                for val in sorted(f.domain):
                    lines.append(f'    value: "{val}"')
                lines.append("  }")
            if f.not_in_environment:
                for env in sorted(f.not_in_environment):
                    lines.append(f'  not_in_environment: "{env}"')
            if f.drift_threshold_l_inf is not None:
                lines.append(f"  drift_comparator {{ infinity_norm {{ threshold: {f.drift_threshold_l_inf} }} }}")
            lines.append("}\n")
        return "\n".join(lines)

    def save_pbtxt(self, path):
        with open(path, "w") as f:
            f.write(self.to_pbtxt())

def generate_statistics(df):
    stats = {"num_rows": len(df), "num_columns": len(df.columns), "features": {}}
    for col in df.columns:
        s = df[col]
        non_null = int(s.count())
        miss_rate = float(s.isna().sum() / len(df))
        if pd.api.types.is_numeric_dtype(s):
            c = s.dropna()
            stats["features"][col] = {
                "type": "NUMERICAL", "non_null_count": non_null, "missing_rate": miss_rate,
                "mean": float(c.mean()) if len(c) else 0, "std": float(c.std()) if len(c) else 0,
                "min": float(c.min()) if len(c) else 0, "max": float(c.max()) if len(c) else 0
            }
        else:
            vc = s.value_counts(normalize=True).to_dict()
            stats["features"][col] = {
                "type": "CATEGORICAL", "non_null_count": non_null, "missing_rate": miss_rate,
                "value_counts": {str(k): float(v) for k, v in vc.items()}
            }
    return stats

def infer_schema(stats):
    schema = DatasetSchema()
    for col, f_stat in stats["features"].items():
        if f_stat["type"] == "NUMERICAL":
            f_type = "INT" if "age" in col or "score" in col or "months" in col or "is_" in col else "FLOAT"
            min_frac = 1.0 if f_stat["missing_rate"] == 0.0 else max(0.5, 1.0 - (f_stat["missing_rate"] * 1.5))
            f = FeatureSchema(name=col, feature_type=f_type, min_fraction=round(min_frac, 2), min_val=f_stat["min"])
        else:
            f = FeatureSchema(name=col, feature_type="STRING", min_fraction=1.0, domain=list(f_stat["value_counts"].keys()))
        schema.add_feature(f)
    return schema

def validate_statistics(stats, schema, environment=None):
    anomalies = []
    actual = stats["features"]
    for feat_name, feat_schema in schema.features.items():
        if environment and environment in feat_schema.not_in_environment:
            continue
        if feat_name not in actual:
            anomalies.append({"feature": feat_name, "type": "Column dropped", "severity": "ERROR", "description": f"Missing feature '{feat_name}'"})
            continue
        act = actual[feat_name]
        pres = 1.0 - act["missing_rate"]
        if pres < feat_schema.min_fraction:
            anomalies.append({"feature": feat_name, "type": "Missing fraction violation", "severity": "ERROR", "description": f"Presence {pres:.2f} < min required {feat_schema.min_fraction:.2f}"})
        if feat_schema.domain and act["type"] == "CATEGORICAL":
            out_of_dom = set(act["value_counts"].keys()) - feat_schema.domain
            if out_of_dom:
                in_mass = sum(act["value_counts"].get(v, 0.0) for v in feat_schema.domain)
                if in_mass < feat_schema.min_domain_mass:
                    anomalies.append({"feature": feat_name, "type": "Unexpected string values", "severity": "ERROR", "description": f"Unexpected values: {sorted(list(out_of_dom))}"})
        if feat_schema.feature_type in ["INT", "FLOAT"] and act["type"] == "NUMERICAL":
            if feat_name in ["debt_to_income_ratio", "annual_income"] and act["min"] < 0:
                anomalies.append({"feature": feat_name, "type": "Out of range value", "severity": "ERROR", "description": f"Negative value {act['min']} in non-negative domain"})
    return anomalies

def compute_drift_and_skew(lhs_stats, rhs_stats, schema):
    report = {"features": {}, "alerts": []}
    lhs_f = lhs_stats["features"]
    rhs_f = rhs_stats["features"]
    for col in lhs_f:
        if col not in rhs_f: continue
        l_stat, r_stat = lhs_f[col], rhs_f[col]
        f_schema = schema.get_feature(col)
        thresh = f_schema.drift_threshold_l_inf if (f_schema and f_schema.drift_threshold_l_inf) else 0.05
        if l_stat["type"] == "CATEGORICAL" and r_stat["type"] == "CATEGORICAL":
            all_keys = set(l_stat["value_counts"].keys()).union(set(r_stat["value_counts"].keys()))
            max_diff = max(abs(l_stat["value_counts"].get(k, 0.0) - r_stat["value_counts"].get(k, 0.0)) for k in all_keys)
            is_alert = max_diff > thresh
            report["features"][col] = {"type": "CATEGORICAL", "l_infinity_norm": round(max_diff, 4), "drift_detected": is_alert}
            if is_alert:
                report["alerts"].append({"feature": col, "metric": "L-infinity", "value": round(max_diff, 4), "threshold": thresh, "message": f"Categorical drift on '{col}': L-inf {max_diff:.4f} > {thresh}"})
    return report
