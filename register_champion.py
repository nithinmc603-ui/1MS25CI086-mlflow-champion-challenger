"""Registers both logged models and sets the aliases `champion` / `challenger` from the recorded winner."""
import json

import mlflow
from mlflow import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_NAME = "sentiment-model"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()
ids = json.load(open("outputs/run_ids.json"))      # written by evaluate.py (no hard-coded run IDs)
winner = client.get_run(ids["compare"]).data.tags["winner"]   # "A" or "B"
print(f"Winner recorded in compare run: Model {winner}")

versions = {}
for key in ("A", "B"):
    run = client.get_run(ids[key])
    mv = mlflow.register_model(f"runs:/{ids[key]}/model", REGISTERED_NAME)
    client.set_model_version_tag(REGISTERED_NAME, mv.version, "hf_model_id", run.data.params["hf_model_id"])
    client.set_model_version_tag(REGISTERED_NAME, mv.version, "run_name", run.info.run_name)
    versions[key] = mv.version

loser = "B" if winner == "A" else "A"
client.set_registered_model_alias(REGISTERED_NAME, "champion", versions[winner])
client.set_registered_model_alias(REGISTERED_NAME, "challenger", versions[loser])
print(f"champion   -> version {versions[winner]} (Model {winner})")
print(f"challenger -> version {versions[loser]} (Model {loser})")
