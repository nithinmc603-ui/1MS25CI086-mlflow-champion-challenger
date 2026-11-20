# Champion–Challenger Sentiment Model Comparison with Hugging Face and MLflow

Student: Nithin · USN: 1MS25CI086 · Course: Skill Lab (MLOps with MLflow)

## 1. Task and models
- **Task:** binary sentiment classification
- **Dataset:** GLUE SST-2 validation split, **100** examples drawn with seed 42 (`data/eval_inputs.csv`)
- **Model A (Champion):** `distilbert-base-uncased-finetuned-sst-2-english` (~67M)
- **Model B (Challenger):** `siebert/sentiment-roberta-large-english` (~355M)
- Both models see the same inputs in the same order. Both output POSITIVE/NEGATIVE, mapped to 1/0.

## 2. How to run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m mlflow server --host 127.0.0.1 --port 5000    # keep running in terminal 1
python evaluate.py                                      # terminal 2: runs both models, logs to MLflow
python register_champion.py                             # registers models, sets aliases
python predict_from_mlflow.py                           # loads models:/sentiment-model@champion
python predict_from_mlflow.py --alias challenger        # stretch: other alias, script unchanged
```
MLflow UI: http://127.0.0.1:5000 (experiment `hf-sentiment-champion-challenger`).
Executed on Google colob (CPU).The MLflow UI screenshots are replaced by equivalent tables from the same MLflow server.

## 3. Champion rule (written BEFORE the final run, not changed afterwards)
- **Primary metric:** macro-F1 (higher is better). Secondary: accuracy (higher), avg latency in ms (lower).
- **Minimum margin:** the Challenger must beat the Champion by **at least 0.01 (1 percentage point)** macro-F1. With only 100 samples, one example is about 1 point, so a smaller gap is indistinguishable from noise.
- **Guardrail:** reject the Challenger if its average latency is more than **10×** the Champion's.
- **Tie-breaker:** if the difference is inside the margin, the **faster** model wins.
- **Direction:** macro-F1, accuracy, precision, recall: higher is better. Latency and total inference time: lower is better.

## 4. Results
*(Copy from `outputs/results_table.csv` after running `evaluate.py`.)*

| Metric | A: DistilBERT | B: RoBERTa-large | B − A |
|---|---|---|---|
| Accuracy | 0.9700 | 0.9500 | -0.0200 |
| Macro-F1 | 0.9691 | 0.9485 | -0.0206 |
| Avg latency (ms) | 67.39 | 636.60 | +569.22 |
| Total inference (s) | 6.74 | 63.66 | +56.92 |

## 5. Decision (150–250 words)
On the same 100 SST-2 validation sentences, Model A (DistilBERT) scored an accuracy of 0.970 and a macro-F1 of 0.9691, while Model B (RoBERTa-large) scored 0.950 and 0.9485. The Challenger is therefore worse by 0.0206 macro-F1, so it does not reach the required +0.01 margin and my rule keeps Model A as the Champion. The latency guardrail did not decide this: Model B was 9.45 times slower (636.6 ms against 67.4 ms per example) and about five times larger (355M against 67M parameters), but it was also less accurate. The gap is only two examples out of 100, so with a small evaluation set it could be noise, and I cannot claim that RoBERTa-large is worse in general. In the five disagreement examples, Model A was correct on four. The errors appear to involve mixed or ambiguous sentiment, such as “fun, flip and terribly hip bit of cinematic entertainment” and short sentences like “oh come on”. The trade-off is clear: the larger model costs much more time and memory without any gain here. My decision is to keep DistilBERT as the Champion and keep RoBERTa-large registered as the Challenger, to be re-tested on a larger evaluation set.

## 6. Limitations and declaration
- Only 100 evaluation examples; a difference of 1–2 points can be noise. Latency was measured on CPU, one example at a time.
- SST-2 sentences are short movie-review snippets, so results may not transfer to other domains.
- **Tools used:** Hugging Face Transformers, MLflow, scikit-learn, matplotlib. An AI assistant (Claude) helped draft the code skeleton; I ran it myself and can explain every line.
