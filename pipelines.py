import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

CSV_FILE = "customer_transactions_10k.csv"

def step_1_ingest_data(file_path=CSV_FILE):
    print("1. DATA INGESTION & VALIDATION")

    df = pd.read_csv(file_path)
    print(f"Loaded '{file_path}': {df.shape[0]} rows, {df.shape[1]} columns.")

    expected_columns = [
        "transaction_id", "timestamp", "customer_age", "annual_income",
        "account_balance", "transaction_amount", "credit_score", "device_type",
        "transaction_channel", "merchant_category", "has_international_card", "is_fraud"
    ]
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Schema Drift Alert! Missing columns: {missing_cols}")
    print("Schema validation: Passed (all expected columns present).")

    null_counts = df.isnull().sum()[df.isnull().sum() > 0]
    print(f"Detected missing values:\n   {null_counts.to_dict()}")
    return df

def step_2_clean_data(df):
    print("2. DATA PREPROCESSING & CLEANING")

    initial_count = len(df)
    cleaned_df = df.drop_duplicates().copy()
    duplicates_removed = initial_count - len(cleaned_df)

    print(f"Deduplication: Removed {duplicates_removed} duplicate records.")
    print(f"Clean dataset size: {len(cleaned_df)} rows.")
    return cleaned_df.reset_index(drop=True)

def step_3_feature_engineering(df):
    print("3. FEATURE ENGINEERING")

    data = df.copy()

    ts = pd.to_datetime(data["timestamp"])
    data["transaction_hour"] = ts.dt.hour
    data["is_night_transaction"] = ((data["transaction_hour"] >= 0) & (data["transaction_hour"] <= 5)).astype(int)
    print("Feature Extraction: Extracted 'transaction_hour' and 'is_night_transaction' from timestamp.")

    data["transaction_to_balance_ratio"] = data["transaction_amount"] / (data["account_balance"] + 1.0)
    print("Feature Creation: Calculated 'transaction_to_balance_ratio'.")

    data = data.drop(columns=["transaction_id", "timestamp"])
    print("Feature Selection: Dropped 'transaction_id' and 'timestamp'.")

    return data

def step_4_split_data(df):
    print("4. DATA SPLITTING - 70% TRAIN / 15% VAL / 15% TEST")

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"Training Set   (70%): {X_train.shape[0]} rows")
    print(f"Validation Set (15%): {X_val.shape[0]} rows")
    print(f"Testing Set    (15%): {X_test.shape[0]} rows")
    return X_train, X_val, X_test, y_train, y_val, y_test

def step_5_create_pipeline(classifier):
    num_cols = [
        "customer_age", "annual_income", "account_balance",
        "transaction_amount", "credit_score", "transaction_to_balance_ratio"
    ]
    cat_cols = [
        "device_type", "transaction_channel", "merchant_category",
        "has_international_card", "is_night_transaction"
    ]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ], remainder="drop")

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    return full_pipeline

def step_6_compare_models(X_train, y_train, X_val, y_val):
    print("5. MODEL SELECTION ON VALIDATION SET")

    models = {
        "Logistic Regression (Linear Baseline)": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42),
        "Gradient Boosting (Ensemble)": GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    }

    best_name = None
    best_f1 = -1.0
    best_pipeline = None

    for name, clf in models.items():
        pipeline = step_5_create_pipeline(clf)
        pipeline.fit(X_train, y_train)

        val_preds = pipeline.predict(X_val)
        val_f1 = f1_score(y_val, val_preds)
        print(f"Model: {name:<38} | Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_name = name
            best_pipeline = pipeline

    print(f"\nBest Selected Model: {best_name} (Val F1: {best_f1:.4f})")
    return best_pipeline

def step_7_tune_hyperparameters(X_train, y_train):
    print("6. HYPERPARAMETER TUNING - RANDOM SEARCH")

    base_pipeline = step_5_create_pipeline(RandomForestClassifier(random_state=42, class_weight="balanced"))

    param_distributions = {
        "classifier__n_estimators": [50, 100, 150],
        "classifier__max_depth": [6, 10, 14, None],
        "classifier__min_samples_split": [2, 5]
    }

    print("Running RandomizedSearchCV (5-fold cross-validation)...")
    search = RandomizedSearchCV(
        base_pipeline,
        param_distributions=param_distributions,
        n_iter=6,
        scoring="f1",
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)

    print(f"Optimal Parameters Found: {search.best_params_}")
    print(f"Cross-Validation F1-Score: {search.best_score_:.4f}")
    return search.best_estimator_

def step_8_evaluate_model(model, X_test, y_test):
    print("7. MODEL EVALUATION & THE ACCURACY PARADOX")

    dummy_preds = np.zeros(len(y_test))
    print("--- Naive 'Guess No Fraud Always' Baseline ---")
    print(f"Naive Accuracy : {accuracy_score(y_test, dummy_preds) * 100:.2f}% (Misleading)")
    print(f"Naive Precision: {precision_score(y_test, dummy_preds, zero_division=0):.4f}")
    print(f"Naive Recall   : {recall_score(y_test, dummy_preds, zero_division=0):.4f}")
    print(f"Naive F1-Score : {f1_score(y_test, dummy_preds, zero_division=0):.4f}")

    test_preds = model.predict(X_test)
    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, zero_division=0)
    rec = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)
    cm = confusion_matrix(y_test, test_preds)

    print("\n--- Tuned Pipeline Performance on Test Set ---")
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1-Score  : {f1:.4f}")

    print(f"\nConfusion Matrix:\n  [TN={cm[0, 0]:<5} FP={cm[0, 1]}]\n  [FN={cm[1, 0]:<5} TP={cm[1, 1]}]")

if __name__ == "__main__":
    print("RUNNING MACHINE LEARNING PIPELINE IN TERMINAL")

    df_raw = step_1_ingest_data()
    df_clean = step_2_clean_data(df_raw)
    df_featured = step_3_feature_engineering(df_clean)
    X_train, X_val, X_test, y_train, y_val, y_test = step_4_split_data(df_featured)
    best_model = step_6_compare_models(X_train, y_train, X_val, y_val)
    tuned_model = step_7_tune_hyperparameters(X_train, y_train)
    step_8_evaluate_model(tuned_model, X_test, y_test)

    print("PIPELINE EXECUTION FINISHED SUCCESSFULLY")