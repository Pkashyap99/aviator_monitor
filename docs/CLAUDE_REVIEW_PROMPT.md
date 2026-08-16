# Claude Review Prompt

You are reviewing an existing Python project that collects Aviatrix crash-game multiplier history and displays a realtime prediction dashboard.

Important constraint:

- Do not suggest hacking, bypassing auth, extracting hidden server seeds, abusing tokens, or manipulating the game.
- Do not tune model thresholds just to improve a known holdout result.
- If the data does not contain predictive signal, say that clearly.
- The goal is to improve honest statistical evaluation, data quality, model calibration, dashboard clarity, and responsible risk reporting.

## Project Summary

The app has:

- `aviator_monitor.py`: connects to a Chrome Aviatrix tab through Playwright/CDP, reads visible multiplier history, live game state, participant count, visible participant table aggregates, provably fair info when visible, and writes CSV files under `data/`.
- `dashboard.py`: serves a realtime local dashboard at `http://127.0.0.1:8765`.
- `ml_features.py`, `ml_auto_retrain.py`: controlled ML retraining and champion/challenger model promotion.
- `edge_audit.py`: statistical audit for possible weak watch patterns using train/holdout validation.
- `data/rounds.csv`: historical multipliers, currently around 14k rows.
- `data/round_context.csv`: participant/context aggregates.
- `data/range_prediction_history.csv` and `data/ml_live_predictions.csv`: prediction tracking.

Recent improvements already added:

- Fast collector polling and page watcher.
- Collector self-healing when multiplier DOM becomes stale or invisible.
- Automatic dashboard ML retraining after enough new rounds.
- Automatic Edge Audit refresh after enough new rounds.
- AI Watch card showing weak candidate patterns and active/current pattern matches.
- Big multiplier tracking for `10x`, `20x`, `50x`, `100x`.

Current honest result:

- Around 14k round records.
- ML champion is still mostly historical-frequency baseline.
- Challenger ML models have not shown enough Brier skill improvement to promote.
- Edge audit finds weak watch candidates but no strong confirmed edge.
- Example weak candidates include low streaks before `10x+`, certain hour buckets, and long gaps before rare large multipliers.
- These are not reliable enough to call a strong prediction edge.

## What I Want Reviewed

Please review the approach and suggest improvements in these areas:

1. Statistical evaluation
   - Is the train/holdout method correct for time-series crash-game data?
   - Should we use walk-forward validation, blocked cross-validation, bootstrap confidence intervals, or Bayesian calibration?
   - How should we avoid false discoveries while testing many patterns?

2. Feature engineering
   - What features are reasonable without cheating or hidden seed access?
   - Examples: recent multiplier buckets, streaks, gaps since `10x/20x/50x/100x`, volatility windows, hour/day, participant count, bet/cashout aggregates.
   - Which features are likely useless and should be removed?

3. Model selection
   - Should we use logistic regression, isotonic calibration, Bayesian hierarchical rates, online learning, random forests, gradient boosting, or only transparent baselines?
   - How should promotion rules be designed so we do not overfit?
   - What metrics should drive promotion: Brier score, log loss, calibration error, precision/recall for rare big multipliers, expected value simulations?

4. Big multiplier prediction
   - Is gap analysis useful for `10x`, `20x`, `50x`, `100x`, or does it create gambler's fallacy?
   - How can the UI show “chance elevated / not elevated” without misleading users?
   - Can rare-event modeling help, or is baseline frequency the best honest estimate?

5. Data quality
   - What checks should be added for duplicate rows, missed rounds, timestamp anomalies, demo/real source mixing, and DOM freeze recovery?
   - How can we tell whether participant/bet context actually adds signal?

6. Dashboard UX
   - How should non-technical users see prediction confidence?
   - What should the app show when there is no reliable signal?
   - How should it explain weak watch candidates, strong candidates, and model-not-beating-baseline?

7. Concrete next steps
   - Give a practical improvement roadmap.
   - Prefer suggestions that can be implemented in Python in this existing repo.
   - Include exact experiments to run and how to decide whether each experiment succeeded or failed.

## Relevant Local Files

Please ask me to paste files or outputs if needed. The most relevant files are:

- `aviator_monitor.py`
- `dashboard.py`
- `edge_audit.py`
- `ml_features.py`
- `ml_auto_retrain.py`
- `data/rounds.csv`
- `data/round_context.csv`
- `data/ml_live_predictions.csv`
- `data/range_prediction_history.csv`
- `data/edge_audit.json`
- `data/ml_report.json`

## Current Commands

Useful commands:

```bash
python3 edge_audit.py --min-sample 80 --top 20
python3 ml_auto_retrain.py --status
python3 ml_auto_retrain.py --live-metrics
python3 ml_auto_retrain.py --run-once --force --reason manual
python3 dashboard.py --host 127.0.0.1 --port 8765
python3 aviator_monitor.py
```

Please respond with:

- Honest assessment of whether better-than-baseline prediction is plausible from this data.
- Statistical risks in the current approach.
- Concrete code/design improvements.
- Experiments worth running next.
- What result would prove the model is genuinely improving rather than just fitting noise.
