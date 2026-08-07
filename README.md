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
- confidence, signal, and edge versus baseline for each target

It also writes:

```text
data/analysis.json
```

Important: crash-game multipliers are normally random. This report estimates historical frequencies and pattern-conditioned probabilities; it does not guarantee future results.

## Realtime dashboard

Run:

```bash
python dashboard.py
```

Open:

```text
http://127.0.0.1:8765
```

The dashboard reads `data/rounds.csv` every second and shows live totals, recent rounds, target probabilities, ensemble next-round probability estimates, confidence, signal, and backtest accuracy.

The dashboard also tracks whether each previous prediction matched the next actual multiplier. It writes:

```text
data/prediction_history.csv
data/prediction_state.json
```

Those results are used to calibrate future prediction edges. If recent tracking accuracy is weak, the model automatically shrinks its edge back toward the historical baseline.
It also learns a small per-target decision margin from recent checked predictions, then uses that margin for future HIGH/LOW calls.
The live model can also switch per-target blend profiles, such as `balanced`, `defensive`, `recent_heavy`, `pattern_heavy`, and `streak_heavy`, based on which profile has matched best in recent tracking.

## Files

- `aviator_monitor.py` - browser collector
- `aviator_analyzer.py` - CSV analysis and probability report
- `dashboard.py` - local realtime dashboard server
- `dashboard/` - dashboard UI files
- `config.example.json` - configuration example
- `requirements.txt`
- `data/rounds.csv` - generated automatically
- `data/prediction_history.csv` - generated prediction audit log
- `data/prediction_state.json` - generated rolling prediction metrics
