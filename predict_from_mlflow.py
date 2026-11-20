"""Loads a model BY ALIAS from the MLflow registry and predicts on brand-new data.

No Hugging Face pipeline is imported or called here; everything goes through MLflow.
    python predict_from_mlflow.py                    # champion
    python predict_from_mlflow.py --alias challenger # stretch goal
"""
import argparse

import mlflow
import pandas as pd
from mlflow import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_NAME = "sentiment-model"

parser = argparse.ArgumentParser()
parser.add_argument("--alias", default="champion")
args = parser.parse_args()

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()
mv = client.get_model_version_by_alias(REGISTERED_NAME, args.alias)
print(f"Loaded alias '{args.alias}': {REGISTERED_NAME} version {mv.version}")
print(f"Hugging Face model ID : {mv.tags.get('hf_model_id')}")
print(f"Source run            : {mv.tags.get('run_name')}")

model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_NAME}@{args.alias}")
new_data = pd.read_csv("data/new_data.csv")
preds = model.predict(new_data[["text"]])
out = preds.copy()
out["expected"] = new_data["expected"].map({1: "POSITIVE", 0: "NEGATIVE"})
out["model_version"] = mv.version
out["hf_model_id"] = mv.tags.get("hf_model_id")
print(out[["text", "label_name", "score", "expected"]].to_string(index=False))
out.to_csv("outputs/new_data_predictions.csv", index=False)
print("Saved outputs/new_data_predictions.csv")
