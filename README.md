# Prompt Safety Evaluation

Local-first evaluation scaffold for Meta Llama Prompt Guard 2 22M, with a compatible placeholder runner for WildGuard migration on a cloud GPU server.

## Data Format

Put CSV files in `data/`. Each file should include:

- `prompt`: input text to classify
- `label`: ground-truth harmful-request label, one of `no`, `yes`, `benign`, `malicious`, `0`, or `1`
- `response`: optional assistant response text, used by WildGuard for refusal and harmful-response classification

Optional columns such as `id`, `source`, and `category` are preserved in prediction outputs.

Prediction files use the shared fields:

- `harmful_request`: whether the user request is harmful, `yes` or `no`
- `refusal`: whether the assistant response is a refusal, `yes`, `no`, or `unknown`
- `harmful_response`: whether the assistant response is harmful, `yes`, `no`, or `unknown`

Prompt Guard only classifies the request, so it writes `unknown` for `refusal` and `harmful_response`.

## Local Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Log in to Hugging Face before downloading gated or agreement-gated models:

```powershell
hf auth login
```

## Run Prompt Guard

```powershell
python run_prompt_guard.py --input data/your_dataset.csv --output outputs/prompt_guard_predictions.csv
python evaluate.py --input outputs/prompt_guard_predictions.csv --output outputs/metrics.json
```

## WildGuard

WildGuard is intended for the cloud GPU server. The code is already prepared for the same CSV format.

On the cloud server:

```bash
git clone <your-repo-url>
cd <repo-name>
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-cloud.txt
hf auth login
```

Run a small smoke test first:

```bash
python run_wildguard.py \
  --input data/your_dataset.csv \
  --output outputs/wildguard_smoke.csv \
  --batch-size 1 \
  --limit 5
```

Then run the full dataset:

```bash
python run_wildguard.py \
  --input data/your_dataset.csv \
  --output outputs/wildguard_predictions.csv \
  --batch-size 4
```

Evaluate:

```bash
python evaluate.py --input outputs/wildguard_predictions.csv --target harmful_request --output outputs/wildguard_harmful_request_metrics.json
python evaluate.py --input outputs/wildguard_predictions.csv --target refusal --output outputs/wildguard_refusal_metrics.json
python evaluate.py --input outputs/wildguard_predictions.csv --target harmful_response --output outputs/wildguard_harmful_response_metrics.json
```

For smaller GPUs, reduce `--batch-size` to `1` and, if needed, reduce `--max-length`.
