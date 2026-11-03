import os
import json
import hashlib
import time
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from pipelines import (
    load_and_prep_data,
    build_session1_pipeline,
    evaluate_model
)

EXPECTED_COLUMNS = [
    "customer_age", "annual_income", "account_balance",
    "transaction_amount", "credit_score", "device_type",
    "transaction_channel", "merchant_category", "has_international_card"
]


def validate_schema(df):
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema Violation: Missing expected columns: {missing_cols}")
    if (df["transaction_amount"] < 0).any():
        raise ValueError("Schema Violation: Negative transaction amounts detected!")
    return True


def mask_pii(df, id_col="transaction_id"):
    if id_col in df.columns:
        df[id_col] = df[id_col].astype(str).apply(
            lambda x: hashlib.sha256(x.encode()).hexdigest()[:16]
        )
    return df


def audit_bias_by_cohort(df, y_true, y_pred, cohort_col="device_type"):
    print(f"\nCohort Bias Audit across: {cohort_col}")
    cohorts = df[cohort_col].unique()
    for c in cohorts:
        idx = (df[cohort_col] == c)
        if idx.sum() > 0:
            sub_f1 = f1_score(y_true[idx], y_pred[idx], zero_division=0)
            sub_acc = accuracy_score(y_true[idx], y_pred[idx])
            print(f"Cohort [{c:>8}] (n={idx.sum():>4}): Accuracy={sub_acc:.4f}, F1-Score={sub_f1:.4f}")


def populate_feature_stores(df, entity_key="customer_id"):
    offline_store = df.copy()
    online_store = {}
    for i, (_, row) in enumerate(df.iterrows()):
        key = row.get(entity_key, f"cust_{i}")
        online_store[key] = {
            "annual_income": row.get("annual_income"),
            "account_balance": row.get("account_balance"),
            "credit_score": row.get("credit_score"),
            "customer_age": row.get("customer_age"),
            "has_international_card": row.get("has_international_card")
        }
    return offline_store, online_store


def register_model(model, model_name, version, metrics, params, stage="Staging", registry_dir="model_registry"):
    os.makedirs(registry_dir, exist_ok=True)
    metadata_file = os.path.join(registry_dir, "metadata.json")

    records = []
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            records = json.load(f)

    artifact_path = os.path.join(registry_dir, f"{model_name}_{version}.joblib")
    joblib.dump(model, artifact_path)

    record = {
        "model_name": model_name,
        "version": version,
        "stage": stage,
        "created_at": datetime.now().isoformat(),
        "artifact_path": artifact_path,
        "metrics": metrics,
        "params": params
    }
    records.append(record)
    with open(metadata_file, "w") as f:
        json.dump(records, f, indent=4)

    return record


def transition_model_stage(model_name, version, new_stage, registry_dir="model_registry"):
    metadata_file = os.path.join(registry_dir, "metadata.json")
    if not os.path.exists(metadata_file):
        return

    with open(metadata_file, "r") as f:
        records = json.load(f)

    for rec in records:
        if rec["model_name"] == model_name and rec["version"] == version:
            rec["stage"] = new_stage
        elif rec["model_name"] == model_name and new_stage == "Production" and rec["stage"] == "Production":
            rec["stage"] = "Archived"

    with open(metadata_file, "w") as f:
        json.dump(records, f, indent=4)


def detect_data_drift(baseline_df, incoming_df, numerical_features, alpha=0.05):
    drift_results = {}
    drift_detected = False

    for feat in numerical_features:
        if feat in baseline_df.columns and feat in incoming_df.columns:
            b_sample = baseline_df[feat].dropna()
            i_sample = incoming_df[feat].dropna()
            
            ks_stat, p_val = stats.ks_2samp(b_sample, i_sample)
            is_drifted = p_val < alpha
            drift_results[feat] = {
                "ks_stat": float(ks_stat),
                "p_value": float(p_val),
                "drift_detected": bool(is_drifted)
            }

            if is_drifted:
                drift_detected = True

    return {"has_drift": drift_detected, "feature_metrics": drift_results}


def detect_concept_drift(champion_metrics, current_metrics, threshold_drop=0.05):
    f1_baseline = champion_metrics.get("f1", 0.0)
    f1_current = current_metrics.get("f1", 0.0)
    diff = f1_baseline - f1_current

    if diff > threshold_drop:
        return True
    return False


def evaluate_ci_cd_gate(challenger_metrics, champion_metrics, latency_ms,
                        min_accuracy=0.90, min_f1=0.20, min_precision=0.15, max_latency_ms=25.0):
    gate_passed = True
    
    acc = challenger_metrics.get("accuracy", 0.0)
    if acc < min_accuracy:
        gate_passed = False

    f1 = challenger_metrics.get("f1", 0.0)
    if f1 < min_f1:
        gate_passed = False

    prec = challenger_metrics.get("precision", 0.0)
    if prec < min_precision:
        gate_passed = False

    if latency_ms > max_latency_ms:
        gate_passed = False

    champ_f1 = champion_metrics.get("f1", 0.0)
    if f1 < champ_f1:
        gate_passed = False

    return gate_passed


def run_automated_retraining_loop(champion_model, champion_metrics, new_data, numerical_cols, categorical_cols):
    target_col = "is_fraud"
    X_new = new_data.drop(columns=[target_col])
    y_new = new_data[target_col]
    X_tr, X_val, y_tr, y_val = train_test_split(X_new, y_new, test_size=0.25, random_state=42, stratify=y_new)

    challenger_pipeline = build_session1_pipeline(numerical_cols, categorical_cols)
    challenger_pipeline.fit(X_tr, y_tr)

    challenger_metrics = evaluate_model(challenger_pipeline, X_val, y_val, model_name="Challenger Model")

    t0 = time.time()
    _ = challenger_pipeline.predict(X_val.iloc[:100])
    latency_ms = ((time.time() - t0) / 100.0) * 1000.0

    promoted = evaluate_ci_cd_gate(challenger_metrics, champion_metrics, latency_ms)

    if promoted:
        print("\nGate Decision: PROMOTED TO PRODUCTION")
        register_model(
            challenger_pipeline,
            model_name="fraud_detection_pipeline",
            version="2.0.0",
            metrics=challenger_metrics,
            params={"n_estimators": 100, "class_weight": "balanced"},
            stage="Production"
        )
        transition_model_stage("fraud_detection_pipeline", "2.0.0", "Production")
        return challenger_pipeline, challenger_metrics
    else:
        print("\nGate Decision: REJECTED")
        return champion_model, champion_metrics


def run_batch_inference(model, batch_df, output_path="batch_scoring_output.csv"):
    preds = model.predict(batch_df)
    probs = model.predict_proba(batch_df)[:, 1]

    results = batch_df.copy()
    results["predicted_fraud"] = preds
    results["fraud_probability"] = probs

    results.to_csv(output_path, index=False)
    return results


def predict_real_time(model, transaction_event, online_store):
    cust_id = transaction_event.get("customer_id", "cust_0")
    online_feats = online_store.get(cust_id)
    if not online_feats:
        online_feats = {
            "annual_income": 50000.0,
            "account_balance": 5000.0,
            "credit_score": 650.0,
            "customer_age": 35.0,
            "has_international_card": 0
        }

    full_payload = {**online_feats, **transaction_event}
    if "customer_id" in full_payload:
        del full_payload["customer_id"]

    input_df = pd.DataFrame([full_payload])

    input_df["tx_to_balance_ratio"] = input_df["transaction_amount"] / (input_df["account_balance"].clip(lower=1.0))
    input_df["tx_to_income_ratio"] = input_df["transaction_amount"] / ((input_df["annual_income"] / 12.0).clip(lower=1.0))

    t0 = time.time()
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0, 1]
    elapsed_ms = (time.time() - t0) * 1000.0

    decision = "FLAGGED FOR REVIEW" if pred == 1 else "APPROVED"
    return {
        "decision": decision,
        "fraud_probability": round(float(prob), 4),
        "inference_time_ms": round(elapsed_ms, 2)
    }


def main():
    data = load_and_prep_data("customer_transactions_10k.csv")
    validate_schema(data)

    target_col = "is_fraud"
    X = data.drop(columns=[target_col])
    y = data[target_col]

    numerical_cols = [
        "customer_age", "annual_income", "account_balance",
        "transaction_amount", "credit_score", "tx_month",
        "tx_day", "tx_hour", "tx_dayofweek",
        "tx_to_balance_ratio", "tx_to_income_ratio"
    ]
    categorical_cols = ["device_type", "transaction_channel", "merchant_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    offline_store, online_store = populate_feature_stores(X_train.head(500))

    champion_pipeline = build_session1_pipeline(numerical_cols, categorical_cols)
    champion_pipeline.fit(X_train, y_train)

    champ_metrics = evaluate_model(champion_pipeline, X_test, y_test, model_name="Champion Model (v1.0.0)")
    register_model(
        champion_pipeline,
        model_name="fraud_detection_pipeline",
        version="1.0.0",
        metrics=champ_metrics,
        params={"n_estimators": 100, "class_weight": "balanced"},
        stage="Production"
    )

    y_test_pred = champion_pipeline.predict(X_test)
    audit_bias_by_cohort(X_test, y_test, y_test_pred, cohort_col="device_type")

    drifted_data = data.copy()
    drifted_data["transaction_amount"] = drifted_data["transaction_amount"] * 1.85
    drifted_data["customer_age"] = (drifted_data["customer_age"] * 0.75).clip(lower=18)

    drift_report = detect_data_drift(
        baseline_df=X_train,
        incoming_df=drifted_data,
        numerical_features=["transaction_amount", "customer_age", "credit_score"]
    )

    if drift_report["has_drift"]:
        prod_model, prod_metrics = run_automated_retraining_loop(
            champion_model=champion_pipeline,
            champion_metrics=champ_metrics,
            new_data=drifted_data,
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols
        )
    else:
        prod_model, prod_metrics = champion_pipeline, champ_metrics

    run_batch_inference(prod_model, X_test.head(100), output_path="batch_scoring_output.csv")

    sample_cust = list(online_store.keys())[0] if online_store else "cust_0"
    sample_event = {
        "customer_id": sample_cust,
        "device_type": "Mobile",
        "transaction_channel": "Online",
        "merchant_category": "Travel",
        "transaction_amount": 420.50,
        "tx_month": 10,
        "tx_day": 27,
        "tx_hour": 15,
        "tx_dayofweek": 1,
        "has_international_card": 1
    }
    real_time_response = predict_real_time(prod_model, sample_event, online_store)
    print("Real-time API Response:", real_time_response)


if __name__ == "__main__":
    main()
