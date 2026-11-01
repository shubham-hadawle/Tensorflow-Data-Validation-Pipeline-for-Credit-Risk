import os
import json
from src.data_loader import load_datasets
from src.data_validator import generate_statistics, infer_schema, validate_statistics, compute_drift_and_skew
from src.model import CreditRiskModel, DataValidationGate

def run():
    print("[1] Loading data...")
    train_df, eval_df, serving_df, drifted_df = load_datasets("data/raw")
    print(f"Loaded Train: {len(train_df)}, Eval: {len(eval_df)}, Serving: {len(serving_df)}, Drifted: {len(drifted_df)}")

    print("[2] Generating statistics & schema...")
    train_stats = generate_statistics(train_df)
    schema = infer_schema(train_stats)
    schema.save_pbtxt("data/schema/baseline_schema.pbtxt")

    print("[3] Detecting anomalies in Eval...")
    eval_stats = generate_statistics(eval_df)
    anom = validate_statistics(eval_stats, schema)
    print(f"Detected {len(anom)} anomalies: {[a['feature'] for a in anom]}")

    print("[4] Relaxing schema constraints...")
    schema.get_feature("loan_intent").min_domain_mass = 0.90
    schema.get_feature("employment_length_years").min_fraction = 0.70

    print("[5] Configuring multi-environment SERVING...")
    schema.default_environment = ["TRAINING", "SERVING"]
    schema.get_feature("is_default").not_in_environment.add("SERVING")
    serving_stats = generate_statistics(serving_df)
    serving_anom = validate_statistics(serving_stats, schema, environment="SERVING")
    print(f"Serving anomalies with environment='SERVING': {len(serving_anom)}")

    print("[6] Checking drift on Drifted batch...")
    schema.get_feature("loan_intent").drift_threshold_l_inf = 0.05
    drifted_stats = generate_statistics(drifted_df)
    drift = compute_drift_and_skew(train_stats, drifted_stats, schema)
    print(f"Triggered {len(drift['alerts'])} drift alerts.")
    schema.save_pbtxt("data/schema/relaxed_schema.pbtxt")

    print("[7] Training ML Credit Risk Classifier...")
    model = CreditRiskModel()
    model.fit(train_df)
    os.makedirs("models", exist_ok=True)
    model.save("models/credit_risk_model.json")
    ev = model.evaluate(eval_df, eval_df["is_default"].values)
    print(f"Model Evaluation: ROC-AUC={ev['roc_auc']}, Acc={ev['accuracy']}")

    print("[8] Running validation gate...")
    gate = DataValidationGate(schema, train_stats)
    res_serve = gate.predict_guarded(model, serving_df.head(200))
    res_drift = gate.predict_guarded(model, drifted_df.head(200))
    print(f"Serving Gate: {res_serve['gate_status']}, Drifted Gate: {res_drift['gate_status']}")
    print("[+] Pipeline finished successfully!")

if __name__ == "__main__":
    run()
