"""Runs Model A (Champion) and Model B (Challenger) on the SAME inputs and logs everything to MLflow.

Usage:
    python -m mlflow server --host 127.0.0.1 --port 5000      # terminal 1
    python evaluate.py                                          # terminal 2
"""
import json
import os
import platform
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from transformers import pipeline

# ----------------------------- configuration -----------------------------
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT = "hf-sentiment-champion-challenger"
STUDENT_USN = "1MS25CI086"
TASK = "sentiment"
DATASET = "glue/sst2 (validation)"
N_SAMPLES = 100
SEED = 42
MAX_LENGTH = 128
BATCH_SIZE = 1          # 1 so that per-example latency is measured honestly
DEVICE = "cpu"

MODEL_A = {"id": "distilbert-base-uncased-finetuned-sst-2-english",
           "short": "distilbert-sst2", "role": "champion", "prefix": "A"}
MODEL_B = {"id": "siebert/sentiment-roberta-large-english",
           "short": "roberta-large", "role": "challenger", "prefix": "B"}

# Champion rule (the SAME rule is written in README.md BEFORE the final run)
PRIMARY_METRIC = "macro_f1"          # higher is better
MARGIN = 0.01                        # challenger must beat champion by >= 1 percentage point
LATENCY_GUARDRAIL = 10.0             # reject challenger if avg latency > 10x champion
# Tie-breaker: if the difference is within the margin, the faster model wins.

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
os.makedirs("data", exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_eval_set():
    path = "data/eval_inputs.csv"
    if os.path.exists(path):          # reuse so both runs always see the same inputs
        return pd.read_csv(path)
    from datasets import load_dataset
    ds = load_dataset("glue", "sst2", split="validation").shuffle(seed=SEED).select(range(N_SAMPLES))
    df = pd.DataFrame({"id": range(len(ds)), "text": ds["sentence"], "label": ds["label"]})
    df.to_csv(path, index=False)
    return df


def run_model(cfg, df):
    pipe = pipeline("sentiment-analysis", model=cfg["id"], device=-1,
                    truncation=True, max_length=MAX_LENGTH)
    pipe("warm up")                                   # exclude first-call overhead from latency
    preds, names, scores, lat = [], [], [], []
    for text in df["text"]:
        t0 = time.perf_counter()
        out = pipe(text)[0]
        lat.append((time.perf_counter() - t0) * 1000)
        name = out["label"].upper()
        names.append(name)
        preds.append(1 if name.startswith("POS") else 0)
        scores.append(round(float(out["score"]), 4))
    res = df.copy()
    res["prediction"], res["label_name"], res["score"] = preds, names, scores
    res["correct"] = (res["prediction"] == res["label"]).astype(int)
    res["latency_ms"] = lat
    return res


def compute_metrics(res):
    y, p = res["label"], res["prediction"]
    return {
        "accuracy": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro"),
        "precision_macro": precision_score(y, p, average="macro", zero_division=0),
        "recall_macro": recall_score(y, p, average="macro", zero_division=0),
        "avg_latency_ms": float(res["latency_ms"].mean()),
        "total_inference_s": float(res["latency_ms"].sum() / 1000),
    }


def log_model_run(cfg, df):
    run_name = f"{cfg['prefix']}_{TASK}_{cfg['short']}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({"hf_model_id": cfg["id"], "task": TASK, "role": cfg["role"],
                           "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE,
                           "eval_samples": len(df), "seed": SEED, "device": DEVICE})
        mlflow.set_tags({"student_usn": STUDENT_USN, "dataset_name": DATASET,
                         "library_versions": f"transformers={transformers.__version__}, "
                                             f"torch={torch.__version__}, mlflow={mlflow.__version__}"})
        res = run_model(cfg, df)
        m = compute_metrics(res)
        mlflow.log_metrics(m)

        # artifacts
        pred_file = f"{OUT}/predictions_{cfg['short']}.csv"
        res.to_csv(pred_file, index=False)
        mlflow.log_artifact(pred_file)
        mlflow.log_artifact("data/eval_inputs.csv")
        mfile = f"{OUT}/metrics_summary_{cfg['short']}.json"
        json.dump({k: round(v, 4) for k, v in m.items()}, open(mfile, "w"), indent=2)
        mlflow.log_artifact(mfile)
        cm = confusion_matrix(res["label"], res["prediction"], labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
        ax.set_xticks([0, 1], ["NEG", "POS"]); ax.set_yticks([0, 1], ["NEG", "POS"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion matrix - {cfg['short']}")
        fig.tight_layout()
        fig.savefig(f"{OUT}/confusion_matrix_{cfg['short']}.png", dpi=150)
        mlflow.log_figure(fig, f"confusion_matrix_{cfg['short']}.png")
        plt.close(fig)

        # model logged through MLflow (pyfunc "models from code")
        kwargs = dict(python_model="hf_sentiment_model.py",
                      model_config={"hf_model_id": cfg["id"], "max_length": MAX_LENGTH,
                                    "batch_size": 8},
                      pip_requirements=["mlflow", "transformers", "torch", "pandas"],
                      metadata={"hf_model_id": cfg["id"], "role": cfg["role"]})
        try:
            mlflow.pyfunc.log_model(name="model", **kwargs)
        except TypeError:             # older MLflow uses artifact_path
            mlflow.pyfunc.log_model(artifact_path="model", **kwargs)
        return run.info.run_id, res, m


def main():
    set_seed(SEED)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    df = load_eval_set()
    print(f"Evaluation set: {len(df)} examples (saved to data/eval_inputs.csv)")

    id_a, res_a, m_a = log_model_run(MODEL_A, df)
    id_b, res_b, m_b = log_model_run(MODEL_B, df)

    # ---------------- decision (follows the rule in README) ----------------
    delta = m_b[PRIMARY_METRIC] - m_a[PRIMARY_METRIC]
    lat_ratio = m_b["avg_latency_ms"] / m_a["avg_latency_ms"]
    if lat_ratio > LATENCY_GUARDRAIL:
        winner, reason = "A", f"guardrail: challenger latency {lat_ratio:.1f}x > {LATENCY_GUARDRAIL}x"
    elif delta >= MARGIN:
        winner, reason = "B", f"challenger beats champion by {delta:.4f} >= margin {MARGIN}"
    elif abs(delta) < MARGIN:
        faster_a = m_a["avg_latency_ms"] <= m_b["avg_latency_ms"]
        winner = "A" if faster_a else "B"
        reason = f"within margin ({delta:+.4f}); tie-breaker -> faster model"
    else:
        winner, reason = "A", f"challenger is worse by {delta:.4f}"
    print(f"Winner: Model {winner} ({reason})")

    with mlflow.start_run(run_name=f"compare_{TASK}") as run:
        mlflow.log_params({"primary_metric": PRIMARY_METRIC, "margin": MARGIN,
                           "latency_guardrail_x": LATENCY_GUARDRAIL, "eval_samples": len(df),
                           "seed": SEED, "model_a": MODEL_A["id"], "model_b": MODEL_B["id"],
                           "run_id_a": id_a, "run_id_b": id_b})
        mlflow.set_tags({"winner": winner, "decision_reason": reason, "student_usn": STUDENT_USN,
                         "dataset_name": DATASET})
        mlflow.log_metrics({"delta_primary_metric": delta, "latency_ratio_b_over_a": lat_ratio})

        # comparison table + bar chart
        keys = ["accuracy", "macro_f1", "precision_macro", "recall_macro"]
        table = pd.DataFrame({"metric": keys + ["avg_latency_ms", "total_inference_s"],
                              "A_champion": [m_a[k] for k in keys + ["avg_latency_ms", "total_inference_s"]],
                              "B_challenger": [m_b[k] for k in keys + ["avg_latency_ms", "total_inference_s"]]})
        table["diff_B_minus_A"] = table["B_challenger"] - table["A_champion"]
        table.round(4).to_csv(f"{OUT}/results_table.csv", index=False)
        mlflow.log_artifact(f"{OUT}/results_table.csv")
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(keys)); w = 0.35
        ax.bar(x - w / 2, [m_a[k] for k in keys], w, label=f"A {MODEL_A['short']}")
        ax.bar(x + w / 2, [m_b[k] for k in keys], w, label=f"B {MODEL_B['short']}")
        ax.set_xticks(x, keys, rotation=15); ax.set_ylim(0.8, 1.0); ax.legend(); ax.set_title("Champion vs Challenger")
        fig.tight_layout(); fig.savefig(f"{OUT}/comparison_bar.png", dpi=150)
        mlflow.log_figure(fig, "comparison_bar.png"); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(res_a["latency_ms"], bins=20, alpha=0.6, label="A")
        ax.hist(res_b["latency_ms"], bins=20, alpha=0.6, label="B")
        ax.set_xlabel("latency per example (ms)"); ax.legend(); ax.set_title("Latency distribution")
        fig.tight_layout(); fig.savefig(f"{OUT}/latency_hist.png", dpi=150)
        mlflow.log_figure(fig, "latency_hist.png"); plt.close(fig)

        # qualitative examples: 5 cases where the models disagree
        dis = res_a[["id", "text", "label"]].copy()
        dis["pred_A"], dis["pred_B"] = res_a["prediction"], res_b["prediction"]
        dis = dis[dis["pred_A"] != dis["pred_B"]].head(5)
        dis["comment"] = ""     # <- write your own short comment for each row
        dis.to_csv(f"{OUT}/disagreements.csv", index=False)
        mlflow.log_artifact(f"{OUT}/disagreements.csv")
        summary = {"model_a": {k: round(v, 4) for k, v in m_a.items()},
                   "model_b": {k: round(v, 4) for k, v in m_b.items()},
                   "delta_primary_metric": round(delta, 4), "winner": winner, "reason": reason}
        json.dump(summary, open(f"{OUT}/metrics_summary.json", "w"), indent=2)
        mlflow.log_artifact(f"{OUT}/metrics_summary.json")
        for f in ["README.md", "requirements.txt", "data/eval_inputs.csv"]:
            if os.path.exists(f):
                mlflow.log_artifact(f)
        summary_run = run.info.run_id

    json.dump({"A": id_a, "B": id_b, "compare": summary_run}, open(f"{OUT}/run_ids.json", "w"), indent=2)
    print(table.round(4).to_string(index=False))
    print("Run IDs saved to outputs/run_ids.json. Next: python register_champion.py")


if __name__ == "__main__":
    main()
