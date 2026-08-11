# Aviator Monitor Setup And Run Guide

This guide is for a new developer who wants to run the Aviatrix collector and realtime dashboard on their own computer.

## 1. Clone And Install

```bash
git clone https://github.com/Pkashyap99/aviator_monitor.git
cd aviator_monitor

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install -r requirements.txt
python3 -m playwright install chromium
```

## 2. Create Local Config

```bash
cp config.example.json config.json
```

Open `config.json` and update:

```json
{
  "game_url": "https://game.aviatrix.bet/...",
  "require_source": "game"
}
```

Use:

- `"game"` to allow demo or real Aviatrix tabs.
- `"real"` to require only the real Aviatrix game tab.
- `"demo"` to require only demo mode.

`config.json` is ignored by Git because each developer may have a different game URL/session.

## 3. Start Chrome For Aviatrix

Open a dedicated Chrome window with remote debugging enabled.

macOS:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --remote-debugging-port=9222 \
  --user-data-dir="$HOME/.aviator-monitor-chrome"
```

Windows PowerShell:

```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" `
  --remote-debugging-port=9222 `
  --user-data-dir="$env:USERPROFILE\.aviator-monitor-chrome"
```

In that Chrome window:

1. Open the Aviatrix game page.
2. Log in manually if needed.
3. Keep the game tab open.

## 4. Start The Collector

Open a second terminal:

```bash
cd aviator_monitor
source .venv/bin/activate
python3 aviator_monitor.py
```

The collector connects to the Chrome window on port `9222`, reads game multipliers, and writes:

```text
data/rounds.csv
data/round_context.csv
data/provably_fair.csv
```

## 5. Start The Dashboard

Open a third terminal:

```bash
cd aviator_monitor
source .venv/bin/activate
python3 dashboard.py --host 127.0.0.1 --port 8765
```

Open the dashboard:

```bash
open http://127.0.0.1:8765
```

If `open` is not available, manually open this URL in a browser:

```text
http://127.0.0.1:8765
```

## Normal Running Layout

Keep these three things running:

```text
Terminal 1: Chrome with --remote-debugging-port=9222
Terminal 2: python3 aviator_monitor.py
Terminal 3: python3 dashboard.py --host 127.0.0.1 --port 8765
```

## Useful ML Commands

Check current champion/retrain status:

```bash
python3 ml_auto_retrain.py --status
```

Check live prediction metrics:

```bash
python3 ml_auto_retrain.py --live-metrics
```

Run a manual controlled retrain:

```bash
python3 ml_auto_retrain.py --run-once --force --reason manual
```

The dashboard also checks retraining automatically when `ml_auto_retrain` is enabled in `config.json`.

## Stop Everything

Press `Ctrl+C` in:

```text
Terminal 2: collector
Terminal 3: dashboard
```

Then close the dedicated Chrome window.

## Notes For Contributors

- Shared dataset snapshots are committed under `data/`.
- Local runtime files such as locks, logs, screenshots, backups, and `config.json` are ignored.
- Do not tune model thresholds only to improve a known holdout result. If the model performs poorly, report that honestly.
- The app records probabilities and accuracy, but it cannot guarantee future multiplier results.
