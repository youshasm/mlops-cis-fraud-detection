"""Kubeflow Pipelines v2 definition for the fraud detection workflow.

The pipeline is intentionally self-contained: each component reads from an input
artifact, writes to an output artifact, and the training/evaluation outputs drive
conditional deployment when AUC-ROC exceeds 0.85.
"""

import argparse
import os

from kfp import compiler, dsl, kubernetes


COMMON_PACKAGES = [
    "pandas",
    "numpy",
    "scikit-learn",
    "xgboost",
    "joblib",
]


@dsl.component(base_image="python:3.10", packages_to_install=COMMON_PACKAGES)
def ingest_component(
    transaction_path: str,
    identity_path: str,
    raw_data: dsl.Output[dsl.Dataset],
) -> None:
    """Merge raw transaction and identity tables."""
    import os

    import pandas as pd

    print("=" * 50)
    print("Kubeflow ingest step")
    print("=" * 50)

    transactions = pd.read_csv(transaction_path)
    identities = pd.read_csv(identity_path)
    merged = transactions.merge(identities, on="TransactionID", how="left")
    os.makedirs(os.path.dirname(raw_data.path), exist_ok=True)
    merged.to_csv(raw_data.path, index=False)
    print(f"Merged data saved to {raw_data.path}")


@dsl.component(base_image="python:3.10", packages_to_install=COMMON_PACKAGES)
def validate_component(
    raw_data: dsl.Input[dsl.Dataset],
    validation_report: dsl.Output[dsl.Dataset],
) -> None:
    """Validate the raw merged dataset."""
    import os

    import pandas as pd

    print("=" * 50)
    print("Kubeflow validate step")
    print("=" * 50)

    frame = pd.read_csv(raw_data.path)
    missing = (frame.isna().mean() * 100).sort_values(ascending=False)
    flagged = missing[missing > 50]
    lines = [
        "Validation report",
        f"Shape: {frame.shape}",
        "Missing value percentages:",
        missing.to_string(),
        "Columns with >50% missing:",
        flagged.to_string() if not flagged.empty else "None",
        f"Class distribution: {frame['isFraud'].value_counts(dropna=False).to_dict() if 'isFraud' in frame.columns else 'unavailable'}",
    ]
    os.makedirs(os.path.dirname(validation_report.path), exist_ok=True)
    with open(validation_report.path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"Validation report saved to {validation_report.path}")


@dsl.component(base_image="python:3.10", packages_to_install=COMMON_PACKAGES)
def preprocess_component(
    raw_data: dsl.Input[dsl.Dataset],
    processed_data: dsl.Output[dsl.Dataset],
) -> None:
    """Clean and impute the raw dataset."""
    import os

    import numpy as np
    import pandas as pd
    from sklearn.impute import KNNImputer, SimpleImputer

    print("=" * 50)
    print("Kubeflow preprocess step")
    print("=" * 50)

    frame = pd.read_csv(raw_data.path)
    sparse_columns = frame.columns[frame.isna().mean() > 0.5]
    frame = frame.drop(columns=sparse_columns)
    numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in frame.columns if column not in numeric_columns]
    if numeric_columns:
        frame[numeric_columns] = SimpleImputer(strategy="median").fit_transform(frame[numeric_columns])
        top_columns = numeric_columns[: min(10, len(numeric_columns))]
        frame[top_columns] = KNNImputer(n_neighbors=5).fit_transform(frame[top_columns])
    if categorical_columns:
        frame[categorical_columns] = SimpleImputer(strategy="most_frequent").fit_transform(frame[categorical_columns])
    os.makedirs(os.path.dirname(processed_data.path), exist_ok=True)
    frame.to_csv(processed_data.path, index=False)
    print(f"Processed data saved to {processed_data.path}")


@dsl.component(base_image="python:3.10", packages_to_install=COMMON_PACKAGES)
def feature_engineer_component(
    processed_data: dsl.Input[dsl.Dataset],
    featured_data: dsl.Output[dsl.Dataset],
) -> None:
    """Create target-encoded and temporal fraud features."""
    import os

    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    print("=" * 50)
    print("Kubeflow feature engineering step")
    print("=" * 50)

    frame = pd.read_csv(processed_data.path)
    target = frame["isFraud"] if "isFraud" in frame.columns else None
    high_cardinality = ["card1", "card2", "addr1", "addr2", "P_emaildomain", "R_emaildomain"]
    if target is not None:
        global_mean = target.mean()
        for column in high_cardinality:
            if column in frame.columns:
                series = frame[column].astype(str)
                stats = frame.groupby(series)["isFraud"].agg(["mean", "count"])
                frame[f"{column}_target_encoded"] = series.map(
                    (stats["mean"] * stats["count"] + global_mean * 20.0) / (stats["count"] + 20.0)
                ).fillna(global_mean)
    if "TransactionDT" in frame.columns:
        transaction_dt = pd.to_numeric(frame["TransactionDT"], errors="coerce").fillna(0)
        frame["transaction_hour"] = ((transaction_dt // 3600) % 24).astype(int)
        frame["transaction_day"] = (transaction_dt // 86400).astype(int)
    numeric_columns = [column for column in frame.select_dtypes(include=[np.number]).columns if column != "isFraud"]
    if numeric_columns:
        frame[numeric_columns] = StandardScaler().fit_transform(frame[numeric_columns])
    os.makedirs(os.path.dirname(featured_data.path), exist_ok=True)
    frame.to_csv(featured_data.path, index=False)
    print(f"Featured data saved to {featured_data.path}")


@dsl.component(base_image="python:3.10", packages_to_install=COMMON_PACKAGES)
def train_component(
    featured_data: dsl.Input[dsl.Dataset],
    model_artifact: dsl.Output[dsl.Model],
    training_report: dsl.Output[dsl.Dataset],
) -> None:
    """Train a best model from the engineered data."""
    import json
    import os

    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from xgboost import XGBClassifier

    print("=" * 50)
    print("Kubeflow train step")
    print("=" * 50)

    frame = pd.read_csv(featured_data.path)
    y = frame["isFraud"].astype(int)
    X = frame.drop(columns=["isFraud"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in X_train.columns if column not in numeric_columns]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_columns),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )

    models = {
        "xgboost": Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(eval_metric="logloss", tree_method="hist", n_estimators=150, random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", preprocessor),
            ("selector", SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))),
            ("model", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]),
    }

    results = []
    fitted_models = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, probabilities)
        results.append({"model": name, "auc_roc": auc_score})
        fitted_models[name] = pipeline

    best_name = max(results, key=lambda item: item["auc_roc"])["model"]
    best_model = fitted_models[best_name]
    os.makedirs(os.path.dirname(model_artifact.path), exist_ok=True)
    joblib.dump(best_model, model_artifact.path)
    with open(training_report.path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Best model: {best_name}")
    print(f"Model saved to {model_artifact.path}")


@dsl.component(base_image="python:3.10", packages_to_install=COMMON_PACKAGES)
def evaluate_component(
    featured_data: dsl.Input[dsl.Dataset],
    model_artifact: dsl.Input[dsl.Model],
    evaluation_report: dsl.Output[dsl.Dataset],
) -> float:
    """Evaluate the trained model and emit the AUC-ROC for deployment gating."""
    import os

    import joblib
    import pandas as pd
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    print("=" * 50)
    print("Kubeflow evaluate step")
    print("=" * 50)

    frame = pd.read_csv(featured_data.path)
    y = frame["isFraud"].astype(int)
    X = frame.drop(columns=["isFraud"])
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = joblib.load(model_artifact.path)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    auc_score = roc_auc_score(y_test, probabilities)
    report = [
        f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}",
        f"Recall: {recall_score(y_test, predictions, zero_division=0):.4f}",
        f"F1: {f1_score(y_test, predictions, zero_division=0):.4f}",
        f"AUC-ROC: {auc_score:.4f}",
        f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}",
    ]
    os.makedirs(os.path.dirname(evaluation_report.path), exist_ok=True)
    with open(evaluation_report.path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report))
    print("\n".join(report))
    return float(auc_score)


@dsl.component(base_image="python:3.10", packages_to_install=COMMON_PACKAGES)
def deploy_component(
    model_artifact: dsl.Input[dsl.Model],
    evaluation_report: dsl.Input[dsl.Dataset],
    deployment_manifest: dsl.Output[dsl.Dataset],
) -> None:
    """Simulate conditional deployment for the best model."""
    import os

    print("=" * 50)
    print("Kubeflow conditional deployment step")
    print("=" * 50)

    with open(evaluation_report.path, "r", encoding="utf-8") as handle:
        report = handle.read()
    os.makedirs(os.path.dirname(deployment_manifest.path), exist_ok=True)
    with open(deployment_manifest.path, "w", encoding="utf-8") as handle:
        handle.write("Deployment triggered for model artifact: " + model_artifact.path + "\n")
        handle.write(report)
    print(f"Deployment manifest written to {deployment_manifest.path}")


@dsl.pipeline(name="fraud-detection-mlops", description="IEEE CIS fraud detection MLOps pipeline")
def fraud_detection_pipeline(
    transaction_path: str = "data/train_transaction.csv",
    identity_path: str = "data/train_identity.csv",
    namespace: str = "fraud-detection",
) -> None:
    """Define the fraud detection pipeline."""
    ingest_task = ingest_component(transaction_path=transaction_path, identity_path=identity_path)
    ingest_task.set_retry(2)
    kubernetes.mount_pvc(
        ingest_task,
        pvc_name="fraud-detection-pvc",
        mount_path="/mnt/shared",
    )

    validate_task = validate_component(raw_data=ingest_task.outputs["raw_data"])
    validate_task.set_retry(2)
    kubernetes.mount_pvc(
        validate_task,
        pvc_name="fraud-detection-pvc",
        mount_path="/mnt/shared",
    )

    preprocess_task = preprocess_component(raw_data=ingest_task.outputs["raw_data"])
    preprocess_task.set_retry(2)
    kubernetes.mount_pvc(
        preprocess_task,
        pvc_name="fraud-detection-pvc",
        mount_path="/mnt/shared",
    )

    feature_task = feature_engineer_component(processed_data=preprocess_task.outputs["processed_data"])
    feature_task.set_retry(2)
    kubernetes.mount_pvc(
        feature_task,
        pvc_name="fraud-detection-pvc",
        mount_path="/mnt/shared",
    )

    train_task = train_component(featured_data=feature_task.outputs["featured_data"])
    train_task.set_retry(2)
    kubernetes.mount_pvc(
        train_task,
        pvc_name="fraud-detection-pvc",
        mount_path="/mnt/shared",
    )

    evaluate_task = evaluate_component(
        featured_data=feature_task.outputs["featured_data"],
        model_artifact=train_task.outputs["model_artifact"],
    )
    evaluate_task.set_retry(2)
    kubernetes.mount_pvc(
        evaluate_task,
        pvc_name="fraud-detection-pvc",
        mount_path="/mnt/shared",
    )

    with dsl.Condition(evaluate_task.outputs["Output"] > 0.85):
        deploy_task = deploy_component(
            model_artifact=train_task.outputs["model_artifact"],
            evaluation_report=evaluate_task.outputs["evaluation_report"],
        )
        deploy_task.set_retry(2)
        kubernetes.mount_pvc(
            deploy_task,
            pvc_name="fraud-detection-pvc",
            mount_path="/mnt/shared",
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compile the Kubeflow fraud pipeline.")
    parser.add_argument(
        "--output-path",
        default=os.path.join("outputs", "pipeline.yaml"),
        help="Where to write the compiled pipeline",
    )
    return parser.parse_args()


def main() -> int:
    """Compile the pipeline package."""
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    compiler.Compiler().compile(pipeline_func=fraud_detection_pipeline, package_path=args.output_path)
    print("=" * 50)
    print(f"Compiled pipeline saved to {args.output_path}")
    print(f"Target namespace: fraud-detection")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
