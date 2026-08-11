# Aviator Monitor

This project connects to an Aviator-style game in a real Chrome browser, reads visible round multipliers, stores them to CSV, and can analyze the saved history.

It intentionally does **not** click the real-money bet or cash-out buttons.

## 1. Install

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
playwright install chromium
```

## 2. Configure

Copy:

```bash
copy config.example.json config.json
```

Edit `config.json`:

- `game_url`: the page you normally open
- `history_selector`: CSS selector matching multiplier/history elements
- `poll_seconds`: how often to scan the page
- `minimum_new_round_gap_seconds`: prevents duplicate captures
- `require_source`: use `game` to accept both demo and real Aviatrix tabs, or `real` to require only real mode

Finding a selector:
1. Open the game in Chrome.
2. Right-click a displayed multiplier, then Inspect.
3. Find the element that contains text such as `1.37x`.
4. Copy a stable CSS selector.
5. Put it in `history_selector`.

## 3. Run

```bash
python aviator_monitor.py
```

A Chromium window opens. Log in yourself, navigate to Aviator, then press Enter in the terminal.

Output is stored in:

```text
data/rounds.csv
```

## Analysis

Run:

```bash
python aviator_analyzer.py
```

The analyzer reads `data/rounds.csv`, prints:

- overall multiplier statistics
- historical probabilities for targets like `>=2.00x`
- an ensemble next-round probability estimate that blends baseline, recent history, pattern matches, and current streak context
- backtest accuracy compared with the historical baseline
- skill versus a simple majority baseline, so broad easy calls do not look better than they really are
- confidence, signal, and edge versus baseline for each target

It also writes:

```text
data/analysis.json
```

Important: crash-game multipliers are normally random. This report estimates historical frequencies and pattern-conditioned probabilities; it does not guarantee future results.

## ML research pipeline

This project also includes a separate machine-learning research pipeline. It is designed to test whether historical multipliers contain a repeatable next-round signal. It does **not** assume the game is predictable.

Install ML dependencies on a MacBook from VS Code terminal:

```bash
cd /Users/kumarprashant/Downloads/aviator_monitor
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Run chronological walk-forward validation:

```bash
python3 ml_backtest.py
```

Train selected models and evaluate the newest untouched holdout block once:

```bash
python3 ml_train.py
```

Print the saved report:

```bash
python3 ml_report.py
```

Generate current next-round ML probability estimates:

```bash
python3 ml_predict.py
```

Useful target-specific backtest:

```bash
python3 ml_backtest.py --target 2 --min-train 5000 --test-size 1000
```

Optional calibration comparison:

```bash
python3 ml_backtest.py --target 2 --calibration uncalibrated,sigmoid,isotonic
```

Artifacts:

```text
data/ml_report.json
data/ml_backtest.csv
data/ml_predictions.json
models/manifest.json
models/target_*.joblib
```

Guardrail: the final holdout is the newest 15-20% of feature rows. It is reserved before model selection and evaluated once. Do not change model settings or thresholds merely to improve that final holdout result after seeing it. If the holdout is poor, the correct conclusion is that no reliable predictive edge has been proven.

## Realtime dashboard

Run:

```bash
python dashboard.py
```

Open:

```text
http://127.0.0.1:8765
```

The dashboard polls `data/rounds.csv` quickly for live updates and shows the next-round estimate, previous prediction result, data source mode, live context, and compact accuracy tracking.
For prediction history, it treats real and demo rows as the same game metric, then also keeps legacy unlabeled rows in the trusted history set.

The dashboard also tracks whether each previous prediction matched the next actual multiplier. It writes:

```text
data/prediction_history.csv
data/prediction_state.json
```

Those results are used to calibrate future prediction edges. If recent tracking accuracy is weak, the model automatically shrinks its edge back toward the historical baseline.
It also learns a small per-target decision margin from recent checked predictions, then uses that margin for future HIGH/LOW calls.
The live model can also switch per-target blend profiles, such as `balanced`, `defensive`, `recent_heavy`, `pattern_heavy`, and `streak_heavy`, based on which profile has matched best in recent tracking.
Range predictions are scored only when the model sees a measurable edge; weak ranges are still displayed as estimates, but are skipped in accuracy tracking.
When recent learned profiles do not beat the majority baseline, the dashboard suppresses the edge and shows `NO CLEAR EDGE` instead of presenting a weak call as actionable.

## Files

- `aviator_monitor.py` - browser collector
- `aviator_analyzer.py` - CSV analysis and probability report
- `ml_features.py` - deterministic leakage-safe feature generation and data-quality reporting
- `ml_backtest.py` - chronological walk-forward validation
- `ml_train.py` - model selection, final holdout evaluation, and model saving
- `ml_predict.py` - current next-round ML probability estimates
- `ml_report.py` - concise report printer
- `dashboard.py` - local realtime dashboard server
- `dashboard/` - dashboard UI files
- `config.example.json` - configuration example
- `requirements.txt`
- `data/rounds.csv` - generated automatically
- `data/prediction_history.csv` - generated prediction audit log
- `data/prediction_state.json` - generated rolling prediction metrics
