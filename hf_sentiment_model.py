"""MLflow pyfunc wrapper around a Hugging Face sentiment pipeline (Lab 9 style).

Logged with "models from code": MLflow stores this file and the model_config,
so the model can be loaded later from runs:/... or models:/... without any
Hugging Face code in the caller's script.
"""
import mlflow
import pandas as pd
from mlflow.pyfunc import PythonModel


class HFSentimentModel(PythonModel):
    def load_context(self, context):
        from transformers import pipeline

        cfg = context.model_config
        self.hf_model_id = cfg["hf_model_id"]
        self.max_length = int(cfg.get("max_length", 128))
        self.batch_size = int(cfg.get("batch_size", 1))
        self.pipe = pipeline(
            "sentiment-analysis",
            model=self.hf_model_id,
            device=-1,  # CPU
            truncation=True,
            max_length=self.max_length,
        )

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            texts = model_input.iloc[:, 0].astype(str).tolist()
        else:
            texts = [str(t) for t in model_input]
        outs = self.pipe(texts, batch_size=self.batch_size)
        labels = [o["label"].upper() for o in outs]
        return pd.DataFrame(
            {
                "text": texts,
                "prediction": [1 if l.startswith("POS") else 0 for l in labels],
                "label_name": labels,
                "score": [round(float(o["score"]), 4) for o in outs],
            }
        )


mlflow.models.set_model(HFSentimentModel())
