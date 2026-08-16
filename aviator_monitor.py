import asyncio
import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import async_playwright


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"

CSV_PATH = DATA_DIR / "rounds.csv"

ROUND_IDS_PATH = DATA_DIR / "round_ids.csv"

PROVABLY_FAIR_PATH = DATA_DIR / "provably_fair.csv"

ROUND_CONTEXT_PATH = DATA_DIR / "round_context.csv"

LOG_PATH = DATA_DIR / "collector.log"

CONFIG_PATH = ROOT / "config.json"

STATE_PATH = DATA_DIR / "state.json"

LOCK_PATH = DATA_DIR / "collector.lock"

BACKUP_DIR = DATA_DIR / "backups"

CSV_HEADERS = [
    "timestamp",
    "multiplier",
    "round_id",
    "source",
]

ROUND_ID_HEADERS = [
    "observed_at",
    "round_id",
    "source",
]

PROVABLY_FAIR_HEADERS = [
    "observed_at",
    "next_seed",
    "server_next_hash",
    "source",
]

ROUND_CONTEXT_HEADERS = [
    "observed_at",
    "round_id",
    "source",
    "game_source",
    "player_count",
    "bet_count",
    "total_bet",
    "avg_bet",
    "max_bet",
    "cashed_out_count",
    "avg_cashout",
    "max_cashout",
    "total_win",
    "max_win",
    "payload_records",
    "context_hash",
]

MAX_NEW_VALUES_PER_SCAN = 150

MAX_STARTUP_RECOVERY_VALUES = 150

MAX_ROUNDS_BACKUPS = 30

PAGE_WATCHER_INTERVAL_MS = 25

DEFAULT_SNAPSHOT_SCAN_SECONDS = 0.2

WATCHER_DRAIN_TIMEOUT_SECONDS = 0.5

PAGE_READ_TIMEOUT_SECONDS = 1.0

SLOW_CONTEXT_READ_TIMEOUT_SECONDS = 1.5

DEFAULT_NO_VISIBLE_RECOVERY_SECONDS = 20

DEFAULT_NO_VISIBLE_RECOVERY_COOLDOWN_SECONDS = 60

DEFAULT_PAGE_RELOAD_TIMEOUT_SECONDS = 12

DEFAULT_PAGE_RELOAD_SETTLE_SECONDS = 3


# =========================================================
# MULTIPLIER REGEX
# =========================================================

MULTIPLIER_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)(?:x)?\s*$")

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

ROUND_ID_KEYS = {
    "roundid",
    "round",
    "gameroundid",
    "betroundid",
    "gameid",
}

BET_AMOUNT_KEYS = {
    "amount",
    "bet",
    "betamount",
    "betvalue",
    "stake",
    "wager",
}

CASHOUT_MULTIPLIER_KEYS = {
    "cashout",
    "cashoutat",
    "cashoutmultiplier",
    "cashoutcoefficient",
    "cashoutcoef",
    "cashoutodd",
    "cashoutodds",
    "coefficient",
    "coef",
    "odd",
    "odds",
}

WIN_AMOUNT_KEYS = {
    "win",
    "won",
    "winamount",
    "payout",
    "profit",
    "cashoutamount",
}

PLAYER_KEYS = {
    "playerid",
    "userid",
    "user",
    "username",
    "nickname",
    "playername",
    "displayname",
    "name",
}

STATUS_KEYS = {
    "status",
    "state",
    "result",
    "betstatus",
}

CONTEXT_RESPONSE_MARKERS = (
    "getuserbets",
    "bets",
    "participants",
    "round",
    "history",
    "cashout",
)

DEFAULT_PARTICIPANT_COUNT_SELECTOR = ".flight-radar-participants-count"

PARTICIPANT_SCAN_SECONDS = 0.5

PARTICIPANT_MIN_WRITE_SECONDS = 1

PARTICIPANT_HEARTBEAT_SECONDS = 30

PARTICIPANT_TABLE_SCAN_SECONDS = 1

PARTICIPANT_TABLE_HEARTBEAT_SECONDS = 2

GAME_STATUS_SCAN_SECONDS = 0.1

GAME_STATUS_MIN_WRITE_SECONDS = 0.25

GAME_STATUS_HEARTBEAT_SECONDS = 5

LIVE_MULTIPLIER_SELECTORS = [
    ".layout-info .game-score",
    ".game-score",
]

ROUND_STATE_PATTERN = re.compile(r"STATE_(?:START|RUN|FINISH)")

WS_TOKEN_ENDPOINT_MARKER = "getwstoken"

WS_TOKEN_CHANNEL_KEYS = {
    "userchannel": "private user",
    "betschannel": "bets",
    "gamechannel": "game state",
    "integrationchannel": "integration",
    "assetschannel": "assets",
    "gameassetschannel": "game assets",
    "participantschannel": "participants",
    "participantsanonymizedchannel": "participants",
    "bonuseschannel": "bonuses",
}


# =========================================================
# HELPERS
# =========================================================

def now_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp_from_millis(milliseconds):
    try:
        return datetime.fromtimestamp(
            float(milliseconds) / 1000
        ).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError):
        return now_string()


def file_timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def log(message):
    DATA_DIR.mkdir(exist_ok=True)

    line = f"[{now_string()}] {message}"

    print(line)

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


async def page_read_with_timeout(awaitable, timeout_seconds, default=None):
    try:
        return await asyncio.wait_for(
            awaitable,
            timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        return default
    except Exception:
        return default


def load_config():
    if not CONFIG_PATH.exists():
        raise SystemExit(
            "config.json not found.\n"
            "Create config.json first."
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_files():

    DATA_DIR.mkdir(exist_ok=True)

    if not ROUND_CONTEXT_PATH.exists():

        with ROUND_CONTEXT_PATH.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                ROUND_CONTEXT_HEADERS
            )

    if not ROUND_IDS_PATH.exists():

        with ROUND_IDS_PATH.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                ROUND_ID_HEADERS
            )

    if not PROVABLY_FAIR_PATH.exists():

        with PROVABLY_FAIR_PATH.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                PROVABLY_FAIR_HEADERS
            )

    if not CSV_PATH.exists():

        with CSV_PATH.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                CSV_HEADERS
            )

        return

    with CSV_PATH.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        missing_headers = [
            header
            for header in CSV_HEADERS
            if header not in fieldnames
        ]

        if not missing_headers:
            return

        rows = list(reader)

    migration_backup = BACKUP_DIR / f"rounds-before-schema-update-{file_timestamp()}.csv"
    BACKUP_DIR.mkdir(exist_ok=True)
    shutil.copy2(
        CSV_PATH,
        migration_backup
    )

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=CSV_HEADERS
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "timestamp": row.get(
                        "timestamp",
                        ""
                    ),
                    "multiplier": row.get(
                        "multiplier",
                        ""
                    ),
                    "round_id": row.get(
                        "round_id",
                        ""
                    ),
                    "source": row.get(
                        "source",
                        ""
                    ),
                }
            )

    log(
        "Updated rounds CSV schema "
        f"({', '.join(missing_headers)}). Backup: {migration_backup.name}"
    )


def load_observed_round_ids():
    if not ROUND_IDS_PATH.exists():
        return set()

    observed = set()

    try:
        with ROUND_IDS_PATH.open(
            "r",
            newline="",
            encoding="utf-8"
        ) as f:
            reader = csv.DictReader(f)

            for row in reader:
                round_id = row.get(
                    "round_id",
                    ""
                )

                if round_id:
                    observed.add(
                        str(round_id)
                    )

    except OSError:
        return set()

    return observed


def append_observed_round_id(round_id, source):
    if round_id is None:
        return

    with ROUND_IDS_PATH.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                now_string(),
                str(round_id),
                source,
            ]
        )


def load_observed_seed_pairs():
    if not PROVABLY_FAIR_PATH.exists():
        return set()

    observed = set()

    try:
        with PROVABLY_FAIR_PATH.open(
            "r",
            newline="",
            encoding="utf-8"
        ) as f:
            reader = csv.DictReader(f)

            for row in reader:
                seed = row.get(
                    "next_seed",
                    ""
                )
                server_hash = row.get(
                    "server_next_hash",
                    ""
                )

                if seed or server_hash:
                    observed.add(
                        (
                            seed,
                            server_hash
                        )
                    )

    except OSError:
        return set()

    return observed


def append_provably_fair_seed(next_seed, server_next_hash, source):
    if not next_seed and not server_next_hash:
        return

    with PROVABLY_FAIR_PATH.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                now_string(),
                next_seed or "",
                server_next_hash or "",
                source,
            ]
        )


def load_observed_context_hashes():
    if not ROUND_CONTEXT_PATH.exists():
        return set()

    observed = set()

    try:
        with ROUND_CONTEXT_PATH.open(
            "r",
            newline="",
            encoding="utf-8"
        ) as f:
            reader = csv.DictReader(f)

            for row in reader:
                context_hash = row.get(
                    "context_hash",
                    ""
                )

                if context_hash:
                    observed.add(
                        context_hash
                    )

    except OSError:
        return set()

    return observed


def normalize_payload_key(key):
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(key).lower()
    )


def ws_token_channel_summary(payload):
    if not isinstance(
        payload,
        dict
    ):
        return None

    detected = []
    labels = []

    for key, value in payload.items():
        normalized = normalize_payload_key(
            key
        )

        if (
            normalized not in WS_TOKEN_CHANNEL_KEYS
            or value in (
                None,
                "",
            )
        ):
            continue

        detected.append(
            str(
                key
            )
        )

        label = WS_TOKEN_CHANNEL_KEYS[normalized]

        if label not in labels:
            labels.append(
                label
            )

    if not detected:
        return None

    return {
        "observed_at": now_string(),
        "detected_keys": sorted(
            detected
        ),
        "labels": sorted(
            labels
        ),
        "total": len(
            detected
        ),
        "secrets_saved": False,
    }


def direct_payload_value(mapping, aliases):
    if not isinstance(mapping, dict):
        return None

    for key, value in mapping.items():
        if normalize_payload_key(key) in aliases:
            return value

    return None


def number_from_value(value):
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        number = float(value)
        return number if number == number else None

    if isinstance(value, str):
        match = NUMBER_RE.search(
            value.replace(
                ",",
                ""
            )
        )

        if not match:
            return None

        try:
            return float(
                match.group(0)
            )
        except ValueError:
            return None

    return None


def direct_number(mapping, aliases):
    return number_from_value(
        direct_payload_value(
            mapping,
            aliases
        )
    )


def extract_payload_round_id(mapping, inherited_round_id=None):
    value = direct_payload_value(
        mapping,
        ROUND_ID_KEYS
    )

    if value is None:
        return inherited_round_id

    round_id = str(
        value
    ).strip()

    return round_id or inherited_round_id


def looks_like_bet_record(mapping):
    if not isinstance(mapping, dict):
        return False

    has_money = any(
        direct_number(
            mapping,
            aliases
        ) is not None
        for aliases in (
            BET_AMOUNT_KEYS,
            CASHOUT_MULTIPLIER_KEYS,
            WIN_AMOUNT_KEYS,
        )
    )

    if has_money:
        return True

    has_player = direct_payload_value(
        mapping,
        PLAYER_KEYS
    ) is not None
    has_status = direct_payload_value(
        mapping,
        STATUS_KEYS
    ) is not None

    return has_player and has_status


def extract_bet_records(value, inherited_round_id=None):
    records = []

    if isinstance(value, dict):
        round_id = extract_payload_round_id(
            value,
            inherited_round_id
        )

        if round_id and looks_like_bet_record(value):
            records.append(
                (
                    round_id,
                    value
                )
            )

        for child in value.values():
            records.extend(
                extract_bet_records(
                    child,
                    round_id
                )
            )

    elif isinstance(value, list):
        for child in value:
            records.extend(
                extract_bet_records(
                    child,
                    inherited_round_id
                )
            )

    return records


def is_cashed_out_record(record, cashout_multiplier, win_amount):
    status = direct_payload_value(
        record,
        STATUS_KEYS
    )

    status_text = (
        str(
            status
        ).lower()
        if status is not None
        else ""
    )

    if status_text:
        if any(
            marker in status_text
            for marker in (
                "pending",
                "lost",
                "cancel",
                "reject",
            )
        ):
            return False

        if any(
            marker in status_text
            for marker in (
                "cash",
                "win",
                "success",
                "paid",
            )
        ):
            return True

    if win_amount is not None and win_amount > 0:
        return True

    if cashout_multiplier is not None and cashout_multiplier >= 1:
        return True

    return any(
        marker in status_text
        for marker in (
            "cash",
            "win",
            "success",
            "paid",
        )
    )


def finite_context_number(value, minimum=0, maximum=1_000_000_000):
    if value is None:
        return None

    if value != value:
        return None

    if value < minimum or value > maximum:
        return None

    return value


def summarize_round_context(payload):
    summaries = {}

    for round_id, record in extract_bet_records(payload):
        summary = summaries.setdefault(
            round_id,
            {
                "round_id": round_id,
                "players": set(),
                "bet_values": [],
                "cashout_values": [],
                "win_values": [],
                "cashed_out_count": 0,
                "payload_records": 0,
            }
        )

        summary["payload_records"] += 1

        player_id = direct_payload_value(
            record,
            PLAYER_KEYS
        )

        if player_id is not None:
            summary["players"].add(
                str(
                    player_id
                )
            )

        bet_amount = finite_context_number(
            direct_number(
                record,
                BET_AMOUNT_KEYS
            )
        )
        cashout_multiplier = finite_context_number(
            direct_number(
                record,
                CASHOUT_MULTIPLIER_KEYS
            ),
            minimum=1,
            maximum=100_000
        )
        win_amount = finite_context_number(
            direct_number(
                record,
                WIN_AMOUNT_KEYS
            )
        )
        cashed_out = is_cashed_out_record(
            record,
            cashout_multiplier,
            win_amount
        )

        if bet_amount is not None:
            summary["bet_values"].append(
                bet_amount
            )

        if cashed_out and cashout_multiplier is not None:
            summary["cashout_values"].append(
                cashout_multiplier
            )

        if cashed_out and win_amount is not None and win_amount > 0:
            summary["win_values"].append(
                win_amount
            )

        if cashed_out:
            summary["cashed_out_count"] += 1

    return list(
        summaries.values()
    )


def average(values):
    if not values:
        return None

    return sum(values) / len(values)


def context_number(value):
    if value is None:
        return ""

    return f"{value:.2f}"


def context_count(value):
    if value is None:
        return ""

    return str(
        int(value)
    )


def context_signature(row):
    payload = {
        key: row.get(
            key,
            ""
        )
        for key in ROUND_CONTEXT_HEADERS
        if key not in (
            "observed_at",
            "context_hash",
        )
    }

    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True
        ).encode(
            "utf-8"
        )
    ).hexdigest()[:16]


def round_context_row(summary, source, game_source):
    bet_values = summary["bet_values"]
    cashout_values = summary["cashout_values"]
    win_values = summary["win_values"]
    player_count = summary.get(
        "player_count_override"
    )

    if player_count is None and summary["players"]:
        player_count = len(
            summary["players"]
        )

    row = {
        "observed_at": now_string(),
        "round_id": summary["round_id"],
        "source": source,
        "game_source": game_source,
        "player_count": context_count(
            player_count
        ),
        "bet_count": context_count(
            len(bet_values) if bet_values else summary["payload_records"]
        ),
        "total_bet": context_number(
            sum(bet_values) if bet_values else None
        ),
        "avg_bet": context_number(
            average(
                bet_values
            )
        ),
        "max_bet": context_number(
            max(bet_values) if bet_values else None
        ),
        "cashed_out_count": context_count(
            summary["cashed_out_count"]
        ),
        "avg_cashout": context_number(
            average(
                cashout_values
            )
        ),
        "max_cashout": context_number(
            max(cashout_values) if cashout_values else None
        ),
        "total_win": context_number(
            sum(win_values) if win_values else None
        ),
        "max_win": context_number(
            max(win_values) if win_values else None
        ),
        "payload_records": context_count(
            summary["payload_records"]
        ),
        "context_hash": "",
    }
    row["context_hash"] = context_signature(
        row
    )
    return row


def append_round_context_summaries(
    summaries,
    source,
    game_source,
    observed_hashes
):
    rows = []

    for summary in summaries:
        row = round_context_row(
            summary,
            source,
            game_source
        )

        if row["context_hash"] in observed_hashes:
            continue

        rows.append(
            row
        )

    if not rows:
        return 0

    with ROUND_CONTEXT_PATH.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=ROUND_CONTEXT_HEADERS
        )

        for row in rows:
            writer.writerow(
                row
            )
            observed_hashes.add(
                row["context_hash"]
            )

        f.flush()
        os.fsync(
            f.fileno()
        )

    return len(rows)


def append_participant_count_context(player_count, game_source):
    observed_at = now_string()
    row = {
        "observed_at": observed_at,
        "round_id": "",
        "source": "flight_radar_dom",
        "game_source": game_source,
        "player_count": context_count(
            player_count
        ),
        "bet_count": "",
        "total_bet": "",
        "avg_bet": "",
        "max_bet": "",
        "cashed_out_count": "",
        "avg_cashout": "",
        "max_cashout": "",
        "total_win": "",
        "max_win": "",
        "payload_records": "1",
        "context_hash": "",
    }
    row["context_hash"] = hashlib.sha256(
        f"{observed_at}:flight_radar_dom:{game_source}:{player_count}".encode(
            "utf-8"
        )
    ).hexdigest()[:16]

    with ROUND_CONTEXT_PATH.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=ROUND_CONTEXT_HEADERS
        )

        writer.writerow(
            row
        )

        f.flush()
        os.fsync(
            f.fileno()
        )


def visible_participants_signature(context, game_source):
    payload = {
        "source": "participants_dom",
        "game_source": game_source,
        "player_count": context.get(
            "visible_rows"
        ),
        "bet_count": context.get(
            "bet_count"
        ),
        "total_bet": context.get(
            "total_bet"
        ),
        "cashed_out_count": context.get(
            "cashed_out_count"
        ),
        "total_win": context.get(
            "total_win"
        ),
    }

    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True
        ).encode(
            "utf-8"
        )
    ).hexdigest()[:16]


def append_visible_participants_context(context, game_source):
    row = {
        "observed_at": now_string(),
        "round_id": "",
        "source": "participants_dom",
        "game_source": game_source,
        "player_count": context_count(
            context.get(
                "visible_rows"
            )
        ),
        "bet_count": context_count(
            context.get(
                "bet_count"
            )
        ),
        "total_bet": context_number(
            context.get(
                "total_bet"
            )
        ),
        "avg_bet": context_number(
            context.get(
                "avg_bet"
            )
        ),
        "max_bet": context_number(
            context.get(
                "max_bet"
            )
        ),
        "cashed_out_count": context_count(
            context.get(
                "cashed_out_count"
            )
        ),
        "avg_cashout": context_number(
            context.get(
                "avg_cashout"
            )
        ),
        "max_cashout": context_number(
            context.get(
                "max_cashout"
            )
        ),
        "total_win": context_number(
            context.get(
                "total_win"
            )
        ),
        "max_win": context_number(
            context.get(
                "max_win"
            )
        ),
        "payload_records": context_count(
            context.get(
                "visible_rows"
            )
        ),
        "context_hash": visible_participants_signature(
            context,
            game_source
        ),
    }

    with ROUND_CONTEXT_PATH.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=ROUND_CONTEXT_HEADERS
        )

        writer.writerow(
            row
        )

        f.flush()
        os.fsync(
            f.fileno()
        )


def append_worker_active_participant_context(
    player_count,
    round_id,
    game_source,
    observed_hashes
):
    if player_count is None:
        return 0

    row = {
        "observed_at": now_string(),
        "round_id": round_id or "",
        "source": "participants_worker_active",
        "game_source": game_source,
        "player_count": context_count(
            player_count
        ),
        "bet_count": "",
        "total_bet": "",
        "avg_bet": "",
        "max_bet": "",
        "cashed_out_count": "",
        "avg_cashout": "",
        "max_cashout": "",
        "total_win": "",
        "max_win": "",
        "payload_records": "1",
        "context_hash": "",
    }
    row["context_hash"] = context_signature(
        row
    )

    if row["context_hash"] in observed_hashes:
        return 0

    with ROUND_CONTEXT_PATH.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=ROUND_CONTEXT_HEADERS
        )

        writer.writerow(
            row
        )

        f.flush()
        os.fsync(
            f.fileno()
        )

    observed_hashes.add(
        row["context_hash"]
    )

    return 1


def first_outcome_round_id(outcome_counts):
    if not isinstance(outcome_counts, dict) or not outcome_counts:
        return ""

    return next(
        iter(
            outcome_counts.keys()
        ),
        ""
    )


def normalize_worker_participant_record(record):
    if not isinstance(record, dict):
        return record

    normalized = dict(
        record
    )
    outcome_id = record.get(
        "outcomeId",
        ""
    )

    if outcome_id:
        normalized["roundId"] = outcome_id

    # Worker events contain global currencies. Prefer EUR fields so aggregate
    # totals use one common unit instead of mixing INR/KRW/EUR/etc.
    if record.get("amountEur") is not None:
        normalized["betAmount"] = record.get(
            "amountEur"
        )
        normalized["amount"] = record.get(
            "amountEur"
        )
    elif record.get("betAmountEur") is not None:
        normalized["betAmount"] = record.get(
            "betAmountEur"
        )
        normalized["amount"] = record.get(
            "betAmountEur"
        )

    if record.get("winAmountEur") is not None:
        normalized["winAmount"] = record.get(
            "winAmountEur"
        )

    return normalized


def extract_worker_participant_events(value):
    events = []

    if isinstance(value, dict):
        if "activeParticipantsEvent" in value:
            events.append(
                (
                    "active",
                    value.get(
                        "activeParticipantsEvent"
                    )
                )
            )

        if "topParticipantsEvent" in value:
            events.append(
                (
                    "top",
                    value.get(
                        "topParticipantsEvent"
                    )
                )
            )

        for child in value.values():
            events.extend(
                extract_worker_participant_events(
                    child
                )
            )

    elif isinstance(value, list):
        for child in value:
            events.extend(
                extract_worker_participant_events(
                    child
                )
            )

    return events


def process_worker_participant_payload(
    payload,
    game_source,
    observed_context_hashes
):
    rows_written = 0

    for event_type, event in extract_worker_participant_events(
        payload
    ):
        if not isinstance(event, dict):
            continue

        total_active = finite_context_number(
            number_from_value(
                event.get(
                    "totalActiveParticipants"
                )
            ),
            maximum=1_000_000
        )
        outcome_counts = event.get(
            "outcomeActiveParticipants",
            {}
        )
        round_id = first_outcome_round_id(
            outcome_counts
        )

        if event_type == "active":
            rows_written += append_worker_active_participant_context(
                total_active,
                round_id,
                game_source,
                observed_context_hashes
            )
            continue

        participants = event.get(
            "participants",
            []
        )

        if not isinstance(participants, list):
            continue

        normalized_payload = {
            "roundId": round_id,
            "participants": [
                normalize_worker_participant_record(
                    participant
                )
                for participant in participants
            ],
        }
        summaries = summarize_round_context(
            normalized_payload
        )

        if total_active is not None:
            for summary in summaries:
                summary["player_count_override"] = total_active

        rows_written += append_round_context_summaries(
            summaries,
            "participants_worker_top",
            game_source,
            observed_context_hashes
        )

    return rows_written


def backup_rounds_csv():
    if not CSV_PATH.exists():
        return

    BACKUP_DIR.mkdir(exist_ok=True)

    backup_path = BACKUP_DIR / f"rounds-{file_timestamp()}.csv"
    shutil.copy2(
        CSV_PATH,
        backup_path
    )

    backups = sorted(
        BACKUP_DIR.glob("rounds-*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for old_backup in backups[MAX_ROUNDS_BACKUPS:]:
        try:
            old_backup.unlink()
        except OSError:
            pass

    log(
        f"Backed up rounds CSV to {backup_path.name}"
    )


def load_recent_round_values(limit=80):
    if not CSV_PATH.exists():
        return []

    rows = []

    try:
        with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    rows.append(
                        round(
                            float(row.get("multiplier", "")),
                            2
                        )
                    )
                except (TypeError, ValueError):
                    continue

    except OSError:
        return []

    return rows[-limit:]


def acquire_collector_lock():
    DATA_DIR.mkdir(exist_ok=True)
    lock_file = LOCK_PATH.open("w", encoding="utf-8")

    try:
        fcntl.flock(
            lock_file,
            fcntl.LOCK_EX | fcntl.LOCK_NB
        )
    except BlockingIOError:
        log(
            "Another collector is already running. Stop it before starting a new one."
        )
        raise SystemExit(1)

    lock_file.write(
        str(
            time.time()
        )
    )
    lock_file.flush()
    return lock_file


def append_round(multiplier, timestamp=None, round_id=None, source=None):

    with CSV_PATH.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                timestamp or now_string(),
                f"{multiplier:.2f}",
                round_id or "",
                source or "",
            ]
        )

        f.flush()
        os.fsync(
            f.fileno()
        )


def load_state():

    if not STATE_PATH.exists():
        return {}

    try:

        with STATE_PATH.open(
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return {}


def save_state(snapshot=None, game_status=None, realtime_channels=None):

    data = load_state()

    if not isinstance(
        data,
        dict
    ):
        data = {}

    data["last_updated"] = now_string()

    if snapshot is not None:
        data["snapshot"] = snapshot

    if game_status is not None:
        data["game_status"] = game_status

    if realtime_channels is not None:
        data["realtime_channels"] = realtime_channels

    tmp_path = STATE_PATH.with_suffix(".json.tmp")

    with tmp_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2
        )

        f.flush()
        os.fsync(
            f.fileno()
        )

    tmp_path.replace(
        STATE_PATH
    )


def phase_from_round_state(round_state, is_preparing=None):
    if round_state == "STATE_RUN":
        return "running"

    if round_state == "STATE_FINISH":
        return "finished"

    if is_preparing:
        return "preparing"

    if round_state == "STATE_START":
        return "starting"

    return None


def parse_bool_text(value):
    if isinstance(value, bool):
        return value

    lowered = str(
        value
    ).strip().lower()

    if lowered == "true":
        return True

    if lowered == "false":
        return False

    return None


def parse_console_game_status_text(text):
    text = str(
        text or ""
    )

    if "GameTimer" not in text and "STATE_" not in text:
        return None

    match = ROUND_STATE_PATTERN.search(
        text
    )
    round_state = match.group(0) if match else None

    is_preparing = None
    preparing_match = re.search(
        r"isPreparing:\s*(true|false)",
        text,
        re.IGNORECASE
    )

    if preparing_match:
        is_preparing = parse_bool_text(
            preparing_match.group(1)
        )

    if (
        "Prepare done" in text
        and round_state is None
    ):
        round_state = "STATE_START"
        is_preparing = False

    phase = phase_from_round_state(
        round_state,
        is_preparing
    )

    if not phase:
        return None

    return {
        "phase": phase,
        "round_state": round_state,
        "is_preparing": is_preparing,
        "source": "console",
    }


def parse_console_game_status_value(value):
    if not isinstance(
        value,
        dict
    ):
        return None

    round_state = value.get(
        "roundState"
    )
    is_preparing = value.get(
        "isPreparing"
    )

    if (
        not round_state
        and is_preparing is None
    ):
        return None

    phase = phase_from_round_state(
        round_state,
        is_preparing
    )

    if not phase:
        return None

    return {
        "phase": phase,
        "round_state": round_state,
        "is_preparing": is_preparing,
        "source": "console",
    }


# =========================================================
# READ MULTIPLIERS
# =========================================================

def normalize_selector_config(selector):
    selectors = [
        ".px-1 > .text-w-60",
        ".bottom-odds-history .text-w-60",
        ".px-1 .text-w-60",
        ".bottom-odds-history",
    ]

    if isinstance(selector, list):
        selectors.extend(selector)
    else:
        selectors.append(selector)

    normalized = []

    for item in selectors:
        if item and item not in normalized:
            normalized.append(item)

    return normalized


async def read_multipliers_once(page, selector):

    texts = await page.evaluate(
        """
        selector => {
          const output = [];
          const nodes = Array.from(document.querySelectorAll(selector));

          for (const node of nodes) {
            const rect = node.getBoundingClientRect();
            const style = window.getComputedStyle(node);

            if (
              rect.width <= 0
              || rect.height <= 0
              || style.visibility === "hidden"
              || style.display === "none"
            ) {
              continue;
            }

            output.push(
              (node.innerText || node.textContent || "").trim()
            );
          }

          return output;
        }
        """,
        selector
    )

    values = []

    for text in texts:

        for piece in re.split(r"\s+", str(text).strip()):

            match = MULTIPLIER_RE.search(piece)

            if not match:
                continue

            try:

                value = float(
                    match.group(1)
                )

                # Basic sanity protection
                if value >= 1:
                    values.append(
                        round(
                            value,
                            2
                        )
                    )

            except ValueError:
                continue

    return values


async def read_multipliers(page, selectors):

    for selector in normalize_selector_config(
        selectors
    ):

        try:

            values = await read_multipliers_once(
                page,
                selector
            )

        except Exception:
            continue

        if values:
            return values

    return []


def normalize_participant_selectors(selector):
    selectors = []

    if isinstance(selector, list):
        selectors.extend(selector)
    else:
        selectors.append(selector)

    selectors.append(
        DEFAULT_PARTICIPANT_COUNT_SELECTOR
    )

    normalized = []

    for item in selectors:
        if item and item not in normalized:
            normalized.append(item)

    return normalized


async def read_participant_count(page, selectors):
    for selector in normalize_participant_selectors(
        selectors
    ):
        try:
            elements = await page.locator(
                selector
            ).all()
        except Exception:
            continue

        for element in elements:
            try:
                if not await element.is_visible():
                    continue

                text = (
                    await element.inner_text()
                ).strip()

            except Exception:
                continue

            value = number_from_value(
                text
            )

            if value is None or value <= 0:
                continue

            return int(
                value
            )

    return None


async def read_visible_participants_table(page):
    try:
        return await page.evaluate(
            """
            () => {
              const clean = (value) => (
                value || ""
              ).replace(/\\s+/g, " ").trim();

              const parseCompactNumber = (value) => {
                const text = clean(value).replace(/,/g, "");
                const match = text.match(/^(\\d+(?:\\.\\d+)?)([KMB])?$/i);

                if (!match) {
                  return null;
                }

                const scale = {
                  K: 1000,
                  M: 1000000,
                  B: 1000000000,
                }[(match[2] || "").toUpperCase()] || 1;

                return Number.parseFloat(match[1]) * scale;
              };

              const parseMultiplier = (value) => {
                const text = clean(value);
                const match = text.match(/^(\\d+(?:\\.\\d+)?)x$/i);

                if (!match) {
                  return null;
                }

                return Number.parseFloat(match[1]);
              };

              const isVisible = (node) => {
                const rect = node.getBoundingClientRect();
                const style = window.getComputedStyle(node);

                return (
                  rect.width > 0
                  && rect.height > 0
                  && style.visibility !== "hidden"
                  && style.display !== "none"
                );
              };

              const hasVisibleParticipantAmount = Array
                .from(document.querySelectorAll(".players-row-amount"))
                .some((node) => isVisible(node));
              const participantsRoot = document.querySelector(".players-table")
                || document.querySelector("[class*='players-table']");

              if (!hasVisibleParticipantAmount && !participantsRoot) {
                return null;
              }

              const summarizeRows = (rows) => {
                if (!rows.length) {
                  return null;
                }

                const sum = (values) => (
                  values.reduce((total, value) => total + value, 0)
                );
                const bets = rows
                  .map((row) => row.bet)
                  .filter((value) => Number.isFinite(value));
                const wins = rows
                  .map((row) => row.win)
                  .filter((value) => Number.isFinite(value));
                const cashouts = rows
                  .map((row) => row.cashout)
                  .filter((value) => Number.isFinite(value));

                if (!bets.length) {
                  return null;
                }

                return {
                  visible_rows: rows.length,
                  bet_count: bets.length,
                  total_bet: bets.length ? sum(bets) : null,
                  avg_bet: bets.length ? sum(bets) / bets.length : null,
                  max_bet: bets.length ? Math.max(...bets) : null,
                  cashed_out_count: cashouts.length,
                  avg_cashout: cashouts.length ? sum(cashouts) / cashouts.length : null,
                  max_cashout: cashouts.length ? Math.max(...cashouts) : null,
                  total_win: wins.length ? sum(wins) : null,
                  max_win: wins.length ? Math.max(...wins) : null,
                };
              };

              const findParticipantRowRoot = (amountNode) => {
                let best = amountNode.parentElement;
                let current = amountNode.parentElement;

                for (let depth = 0; current && depth < 8; depth += 1) {
                  const amountCount = current
                    .querySelectorAll(".players-row-amount").length;
                  const rect = current.getBoundingClientRect();

                  if (amountCount === 1 && rect.height <= 140) {
                    best = current;
                    current = current.parentElement;
                    continue;
                  }

                  break;
                }

                return best || amountNode.parentElement || amountNode;
              };

              const visibleCellItems = (root) => (
                Array.from(root.querySelectorAll("*"))
                  .filter((node) => isVisible(node))
                  .map((node) => {
                    const rect = node.getBoundingClientRect();

                    return {
                      node,
                      text: clean(node.innerText || node.textContent || ""),
                      x: rect.left,
                      y: rect.top + rect.height / 2,
                    };
                  })
                  .filter((item) => item.text && item.text.length <= 80)
              );

              const firstParsedFromSelectors = (root, selectors, parser) => {
                for (const selector of selectors) {
                  const nodes = Array.from(root.querySelectorAll(selector));

                  for (const node of nodes) {
                    if (!isVisible(node)) {
                      continue;
                    }

                    const value = parser(
                      node.innerText || node.textContent || ""
                    );

                    if (Number.isFinite(value)) {
                      return value;
                    }
                  }
                }

                return null;
              };

              const readClassBasedParticipantRows = () => {
                const amountNodes = Array
                  .from(
                    (participantsRoot || document).querySelectorAll(
                      ".players-row-amount"
                    )
                  )
                  .filter((node) => isVisible(node));
                const rows = [];
                const seen = new Set();

                for (const amountNode of amountNodes) {
                  const bet = parseCompactNumber(
                    amountNode.innerText || amountNode.textContent || ""
                  );

                  if (!Number.isFinite(bet)) {
                    continue;
                  }

                  const root = amountNode.closest(".players-row")
                    || findParticipantRowRoot(amountNode);
                  const rootRect = root.getBoundingClientRect();
                  const amountRect = amountNode.getBoundingClientRect();
                  const key = [
                    Math.round(rootRect.top),
                    Math.round(rootRect.left),
                    Math.round(amountRect.top),
                    Math.round(amountRect.left),
                    bet,
                  ].join(":");

                  if (seen.has(key)) {
                    continue;
                  }

                  seen.add(key);

                  const cashout = firstParsedFromSelectors(
                    root,
                    [
                      ".players-row-win-odd",
                      ".players-row-coefficient",
                      ".players-row-multiplier",
                      ".players-row-cashout",
                      "[class*='win-odd']",
                      "[class*='coefficient']",
                      "[class*='multiplier']",
                      "[class*='cashout']",
                    ],
                    parseMultiplier
                  );
                  const classWin = firstParsedFromSelectors(
                    root,
                    [
                      ".players-row-win-amount",
                      ".players-row-win",
                      ".players-row-winning",
                      ".players-row-payout",
                      ".players-row-profit",
                      "[class*='win-amount']",
                      "[class*='winning']",
                      "[class*='payout']",
                      "[class*='profit']",
                    ],
                    parseCompactNumber
                  );
                  const rowItems = visibleCellItems(root);
                  const fallbackCashouts = rowItems
                    .map((item) => ({
                      value: parseMultiplier(item.text),
                      x: item.x,
                    }))
                    .filter((item) => Number.isFinite(item.value))
                    .sort((left, right) => left.x - right.x);
                  const fallbackWins = rowItems
                    .filter((item) => (
                      item.node !== amountNode
                      && !item.node.contains(amountNode)
                      && !amountNode.contains(item.node)
                      && item.x > amountRect.left + amountRect.width / 2
                    ))
                    .map((item) => ({
                      value: parseCompactNumber(item.text),
                      x: item.x,
                    }))
                    .filter((item) => Number.isFinite(item.value))
                    .sort((left, right) => left.x - right.x);
                  const resolvedCashout = Number.isFinite(cashout)
                    ? cashout
                    : (
                      fallbackCashouts.length
                        ? fallbackCashouts[0].value
                        : null
                    );
                  const resolvedWin = Number.isFinite(resolvedCashout)
                    ? (
                      Number.isFinite(classWin)
                        ? classWin
                        : (
                          fallbackWins.length
                            ? fallbackWins[fallbackWins.length - 1].value
                            : null
                        )
                    )
                    : null;

                  rows.push({
                    bet,
                    win: resolvedWin,
                    cashout: resolvedCashout,
                  });
                }

                return rows;
              };

              const classBasedSummary = summarizeRows(
                readClassBasedParticipantRows()
              );

              if (classBasedSummary) {
                return classBasedSummary;
              }

              const scanRoot = participantsRoot || document.body;
              const nodes = Array.from(scanRoot.querySelectorAll("*"));
              const visibleTextNodes = [];

              for (const node of nodes) {
                if (!isVisible(node)) {
                  continue;
                }

                const text = clean(node.innerText || node.textContent || "");

                if (!text || text.length > 80) {
                  continue;
                }

                const rect = node.getBoundingClientRect();

                visibleTextNodes.push({
                  text,
                  x: rect.left,
                  y: rect.top + rect.height / 2,
                  top: rect.top,
                  height: rect.height,
                });
              }

              const placeBetY = visibleTextNodes
                .filter((item) => /PLACE BET/i.test(item.text))
                .map((item) => item.top)
                .sort((left, right) => left - right)[0] || window.innerHeight;

              const tableItems = visibleTextNodes
                .filter((item) => item.top < placeBetY - 8)
                .filter((item) => (
                  parseCompactNumber(item.text) !== null
                  || parseMultiplier(item.text) !== null
                  || item.text === "-"
                ));

              const rowMap = new Map();

              for (const item of tableItems) {
                const rowKey = String(Math.round(item.y / 28) * 28);
                const row = rowMap.get(rowKey) || [];
                row.push(item);
                rowMap.set(rowKey, row);
              }

              const rows = [];

              for (const row of rowMap.values()) {
                const ordered = row
                  .slice()
                  .sort((left, right) => left.x - right.x);
                const moneyValues = [];
                const multipliers = [];

                for (const item of ordered) {
                  const money = parseCompactNumber(item.text);
                  const multiplier = parseMultiplier(item.text);

                  if (money !== null) {
                    moneyValues.push(money);
                  }

                  if (multiplier !== null) {
                    multipliers.push(multiplier);
                  }
                }

                if (!moneyValues.length) {
                  continue;
                }

                // Exclude quick-bet preset rows such as 500 / 2K / 5K / 20K.
                if (moneyValues.length >= 4 && !multipliers.length) {
                  continue;
                }

                const cashout = multipliers.length ? multipliers[0] : null;

                rows.push({
                  bet: moneyValues[0],
                  win: cashout !== null && moneyValues.length > 1
                    ? moneyValues[moneyValues.length - 1]
                    : null,
                  cashout,
                });
              }

              return summarizeRows(rows);
            }
            """
        )
    except Exception:
        return None


async def read_multipliers_reliably(
    page,
    selectors,
    attempts=4,
    delay_seconds=0.25
):

    for attempt in range(attempts):

        values = await read_multipliers(
            page,
            selectors
        )

        if values:
            return values

        if attempt < attempts - 1:

            await asyncio.sleep(
                delay_seconds
            )

    return []


async def install_history_watcher(
    page,
    selectors,
    initial_snapshot
):
    selector_list = normalize_selector_config(
        selectors
    )

    await page.evaluate(
        """
        ({ selectors, initialSnapshot, intervalMs, maxNewValues }) => {
          const idAttributes = [
            "data-round-id",
            "data-game-id",
            "data-id",
            "data-round",
            "data-history-id",
            "data-bet-round-id",
            "id",
          ];

          const extractRoundId = (node) => {
            let current = node;

            for (let depth = 0; current && depth < 4; depth += 1) {
              for (const attribute of idAttributes) {
                const value = current.getAttribute(attribute);

                if (value && /\\d/.test(value)) {
                  return value;
                }
              }

              current = current.parentElement;
            }

            return null;
          };

            const readValues = () => {
            const values = [];
            const seen = new Set();

            for (const selector of selectors) {
              const nodes = Array.from(document.querySelectorAll(selector));

              for (const node of nodes) {
                const rect = node.getBoundingClientRect();
                const style = window.getComputedStyle(node);

                if (
                  rect.width <= 0
                  || rect.height <= 0
                  || style.visibility === "hidden"
                  || style.display === "none"
                ) {
                  continue;
                }

                const text = (node.innerText || node.textContent || "").trim();
                const parts = text.split(/\\s+/).filter(Boolean);

                for (const part of parts) {
                  const match = part.match(/^(\\d+(?:\\.\\d+)?)(?:x)?$/i);

                  if (!match) {
                    continue;
                  }

                  const value = Number.parseFloat(match[1]);

                  if (!Number.isFinite(value) || value < 1) {
                    continue;
                  }

                  const rounded = Math.round(value * 100) / 100;
                  const key = `${seen.size}:${rounded}`;
                  seen.add(key);
                  values.push({
                    value: rounded,
                    roundId: extractRoundId(node),
                  });
                }
              }

              if (values.length) {
                return values;
              }
            }

            return [];
          };

          const findNewValues = (oldSnapshot, newSnapshot) => {
            if (!oldSnapshot.length || !newSnapshot.length) {
              return [];
            }

            const maxShift = Math.min(newSnapshot.length, 200);
            const anchors = [80, 50, 30, 20, 12, 8, 5, 3];
            const snapshotValue = (item) => (
              typeof item === "number" ? item : item.value
            );

            for (const anchorLength of anchors) {
              if (oldSnapshot.length < anchorLength) {
                continue;
              }

              const oldAnchor = oldSnapshot.slice(0, anchorLength);

              for (let shift = 1; shift <= maxShift; shift += 1) {
                if (shift + anchorLength > newSnapshot.length) {
                  break;
                }

                const newAnchor = newSnapshot.slice(shift, shift + anchorLength);

                if (
                  oldAnchor.every(
                    (item, index) => snapshotValue(item) === snapshotValue(newAnchor[index])
                  )
                ) {
                  return newSnapshot.slice(0, shift);
                }
              }
            }

            return [];
          };

          if (window.__aviatorMonitorWatcher?.timer) {
            clearInterval(window.__aviatorMonitorWatcher.timer);
          }

          if (window.__aviatorMonitorWatcher?.observer) {
            window.__aviatorMonitorWatcher.observer.disconnect();
          }

          window.__aviatorMonitorWatcher = {
            queue: [],
            snapshot: Array.isArray(initialSnapshot)
              ? initialSnapshot.map((value) => (
                  typeof value === "number" ? { value, roundId: null } : value
                ))
              : [],
            lastVisibleAt: Date.now(),
            reads: 0,
            misses: 0,
            timer: null,
            observer: null,
            pendingMutationTick: false,
          };

          const tick = () => {
            const state = window.__aviatorMonitorWatcher;
            const detectedAt = Date.now();
            const current = readValues();
            state.reads += 1;

            if (!current.length) {
              state.misses += 1;
              return;
            }

            state.lastVisibleAt = Date.now();

            if (!state.snapshot.length) {
              state.snapshot = current;
              return;
            }

            const newValues = findNewValues(state.snapshot, current);

            if (newValues.length && newValues.length <= maxNewValues) {
              state.queue.push(
                ...newValues
                  .slice()
                  .reverse()
                  .map((item) => ({
                    value: typeof item === "number" ? item : item.value,
                    roundId: typeof item === "number" ? null : item.roundId,
                    detectedAt,
                  }))
              );
            }

            state.snapshot = current;
          };

          const scheduleImmediateTick = () => {
            const state = window.__aviatorMonitorWatcher;

            if (!state || state.pendingMutationTick) {
              return;
            }

            state.pendingMutationTick = true;

            window.setTimeout(() => {
              const latestState = window.__aviatorMonitorWatcher;

              if (!latestState) {
                return;
              }

              latestState.pendingMutationTick = false;
              tick();
            }, 10);
          };

          window.__aviatorMonitorWatcher.observer = new MutationObserver(
            scheduleImmediateTick
          );

          window.__aviatorMonitorWatcher.observer.observe(
            document.body,
            {
              childList: true,
              subtree: true,
              characterData: true,
            }
          );

          tick();
          window.__aviatorMonitorWatcher.timer = setInterval(tick, intervalMs);
        }
        """,
        {
            "selectors": selector_list,
            "initialSnapshot": initial_snapshot,
            "intervalMs": PAGE_WATCHER_INTERVAL_MS,
            "maxNewValues": MAX_NEW_VALUES_PER_SCAN,
        }
    )

    log(
        f"Installed page watcher at {PAGE_WATCHER_INTERVAL_MS}ms."
    )


async def drain_history_watcher(page):
    try:
        return await page.evaluate(
            """
            () => {
              const state = window.__aviatorMonitorWatcher;

              if (!state) {
                return null;
              }

              const queue = state.queue.slice();
              state.queue = [];

              return {
                queue,
                snapshot: (state.snapshot || []).map((item) => (
                  typeof item === "number" ? item : item.value
                )),
                reads: state.reads || 0,
                misses: state.misses || 0,
                lastVisibleAt: state.lastVisibleAt || null,
              };
            }
            """
        )
    except Exception:
        return None


async def install_game_status_watcher(page):
    await page.evaluate(
        """
        ({ selectors, intervalMs }) => {
          if (window.__aviatorMonitorGameStatus?.timer) {
            clearInterval(window.__aviatorMonitorGameStatus.timer);
          }

          if (window.__aviatorMonitorGameStatus?.observer) {
            window.__aviatorMonitorGameStatus.observer.disconnect();
          }

          const readLiveMultiplier = () => {
            for (const selector of selectors) {
              const node = document.querySelector(selector);

              if (!node) {
                continue;
              }

              const rect = node.getBoundingClientRect();
              const style = window.getComputedStyle(node);

              if (
                rect.width <= 0
                || rect.height <= 0
                || style.visibility === "hidden"
                || style.display === "none"
              ) {
                continue;
              }

              const chars = Array.from(
                node.querySelectorAll("[data-char]")
              ).map((item) => item.getAttribute("data-char") || "").join("");
              const text = (chars || node.innerText || node.textContent || "").trim();
              const match = text.match(/(\\d+(?:\\.\\d+)?)\\s*x?/i);

              if (!match) {
                continue;
              }

              const value = Number.parseFloat(match[1]);

              if (!Number.isFinite(value) || value < 1) {
                continue;
              }

              return Math.round(value * 100) / 100;
            }

            return null;
          };

          window.__aviatorMonitorGameStatus = {
            liveMultiplier: null,
            phase: null,
            source: null,
            updatedAt: null,
            reads: 0,
            misses: 0,
            timer: null,
            observer: null,
            pendingMutationTick: false,
          };

          const tick = () => {
            const state = window.__aviatorMonitorGameStatus;
            const multiplier = readLiveMultiplier();
            state.reads += 1;
            state.updatedAt = Date.now();

            if (multiplier === null) {
              state.misses += 1;
              state.liveMultiplier = null;
              return;
            }

            state.liveMultiplier = multiplier;
            state.phase = "running";
            state.source = "live_score_dom";
          };

          const scheduleImmediateTick = () => {
            const state = window.__aviatorMonitorGameStatus;

            if (!state || state.pendingMutationTick) {
              return;
            }

            state.pendingMutationTick = true;

            window.setTimeout(() => {
              const latestState = window.__aviatorMonitorGameStatus;

              if (!latestState) {
                return;
              }

              latestState.pendingMutationTick = false;
              tick();
            }, 10);
          };

          window.__aviatorMonitorGameStatus.observer = new MutationObserver(
            scheduleImmediateTick
          );

          window.__aviatorMonitorGameStatus.observer.observe(
            document.body,
            {
              childList: true,
              subtree: true,
              characterData: true,
            }
          );

          tick();
          window.__aviatorMonitorGameStatus.timer = setInterval(tick, intervalMs);
        }
        """,
        {
            "selectors": LIVE_MULTIPLIER_SELECTORS,
            "intervalMs": int(
                GAME_STATUS_SCAN_SECONDS * 1000
            ),
        }
    )

    log(
        "Installed game status watcher."
    )


async def read_game_status_watcher(page):
    try:
        return await page.evaluate(
            """
            () => {
              const state = window.__aviatorMonitorGameStatus;

              if (!state) {
                return null;
              }

              return {
                phase: state.phase || null,
                liveMultiplier: state.liveMultiplier,
                source: state.source,
                updatedAt: state.updatedAt,
                reads: state.reads || 0,
                misses: state.misses || 0,
              };
            }
            """
        )
    except Exception:
        return None


def extract_round_ids(value):
    round_ids = []

    if isinstance(value, dict):

        for key, child in value.items():

            if (
                normalize_payload_key(
                    key
                ) in ROUND_ID_KEYS
                and isinstance(
                    child,
                    (
                        str,
                        int
                    )
                )
            ):

                round_ids.append(
                    str(child)
                )

            round_ids.extend(
                extract_round_ids(
                    child
                )
            )

    elif isinstance(value, list):

        for child in value:

            round_ids.extend(
                extract_round_ids(
                    child
                )
            )

    return round_ids


def payload_text_may_have_round_context(text):
    lowered = text.lower()

    return (
        any(
            marker in lowered
            for marker in CONTEXT_RESPONSE_MARKERS
        )
        or "roundid" in lowered
    )


def decode_frame_payload(payload):
    if payload is None:
        return ""

    if isinstance(payload, bytes):
        try:
            return payload.decode(
                "utf-8",
                errors="ignore"
            )
        except Exception:
            return ""

    return str(
        payload
    )


def json_payloads_from_text(text):
    payloads = []
    decoder = json.JSONDecoder()

    for chunk in str(
        text
    ).split(
        "\x1e"
    ):
        stripped = chunk.strip()

        if not stripped:
            continue

        starts = [
            index
            for index in (
                stripped.find("{"),
                stripped.find("["),
            )
            if index >= 0
        ]

        for start in sorted(
            starts
        ):
            try:
                payload, _ = decoder.raw_decode(
                    stripped[start:]
                )
            except json.JSONDecodeError:
                continue

            payloads.append(
                payload
            )
            break

    return payloads


def response_may_have_round_context(response_url):
    lowered = response_url.lower()

    return any(
        marker in lowered
        for marker in CONTEXT_RESPONSE_MARKERS
    )


def response_source_label(response_url):
    parsed = urlparse(
        response_url
    )
    path_parts = [
        part
        for part in parsed.path.split(
            "/"
        )
        if part
    ]

    if path_parts:
        return path_parts[-1][:80]

    return (
        parsed.netloc
        or "network"
    )[:80]


def request_payload_from_response(response):
    request = getattr(
        response,
        "request",
        None
    )

    if request is None:
        return None

    for attribute in (
        "post_data_json",
        "post_data",
    ):
        try:
            value = getattr(
                request,
                attribute,
                None
            )

            if callable(value):
                value = value()

        except Exception:
            continue

        if not value:
            continue

        if isinstance(value, (dict, list)):
            return value

        try:
            return json.loads(
                value
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            continue

    return None


def attach_request_round_id(payload, request_payload):
    if not isinstance(request_payload, dict):
        return payload

    round_id = extract_payload_round_id(
        request_payload
    )

    if not round_id:
        return payload

    if isinstance(payload, dict):
        if extract_payload_round_id(
            payload
        ):
            return payload

        wrapped = {
            "roundId": round_id,
            **payload,
        }

        return wrapped

    if isinstance(payload, list):
        return {
            "roundId": round_id,
            "items": payload,
        }

    return payload


def process_context_payload(
    payload,
    source_label,
    game_source,
    observed_round_ids,
    observed_context_hashes,
    collect_round_context=True
):
    new_count = 0

    for round_id in extract_round_ids(
        payload
    ):

        if round_id in observed_round_ids:
            continue

        observed_round_ids.add(
            round_id
        )

        append_observed_round_id(
            round_id,
            source_label
        )

        new_count += 1

    if new_count:
        log(
            f"Observed {new_count} new game round IDs from {source_label}."
        )

    context_count = 0

    if collect_round_context:
        summaries = summarize_round_context(
            payload
        )

        context_count = append_round_context_summaries(
            summaries,
            source_label,
            game_source,
            observed_context_hashes
        )

        if context_count:
            log(
                "Captured "
                f"{context_count} round context aggregate rows from {source_label}."
            )

    return new_count, context_count


async def install_round_id_observer(page, collect_round_context=True):
    observed_round_ids = load_observed_round_ids()
    observed_context_hashes = load_observed_context_hashes()
    websocket_urls = {}

    async def handle_response(response):
        response_url = response.url
        is_ws_token_response = (
            WS_TOKEN_ENDPOINT_MARKER
            in response_url.lower()
        )
        source_label = response_source_label(
            response_url
        )

        if (
            not is_ws_token_response
            and not response_may_have_round_context(
                response_url
            )
        ):
            return

        try:
            payload = await response.json()
        except Exception:
            return

        if is_ws_token_response:
            channel_summary = ws_token_channel_summary(
                payload
            )

            if channel_summary:
                save_state(
                    realtime_channels=channel_summary
                )
                log(
                    "Detected realtime channels: "
                    f"{', '.join(channel_summary['labels'])}. "
                    "Token values were not saved."
                )

        if not response_may_have_round_context(
            response_url
        ):
            return

        payload = attach_request_round_id(
            payload,
            request_payload_from_response(
                response
            )
        )

        process_context_payload(
            payload,
            source_label,
            page_source(
                page.url
            ),
            observed_round_ids,
            observed_context_hashes,
            collect_round_context
        )

    def handle_websocket_frame(event):
        payload_text = decode_frame_payload(
            event.get(
                "response",
                {}
            ).get(
                "payloadData",
                ""
            )
        )

        if not payload_text_may_have_round_context(
            payload_text
        ):
            return

        request_id = event.get(
            "requestId",
            ""
        )
        websocket_url = websocket_urls.get(
            request_id,
            ""
        )
        source_label = "websocket"

        if websocket_url:
            source_label = "ws:" + response_source_label(
                websocket_url
            )

        for payload in json_payloads_from_text(
            payload_text
        ):
            process_context_payload(
                payload,
                source_label,
                page_source(
                    page.url
                ),
                observed_round_ids,
                observed_context_hashes,
                collect_round_context
            )

    worker_sessions = set()
    attached_worker_targets = set()
    worker_message_counter = {
        "next": 1,
    }
    worker_stats = {
        "rows": 0,
        "last_log": 0,
    }

    def is_realtime_worker_target(target_info):
        if not isinstance(target_info, dict):
            return False

        return (
            target_info.get("type") == "worker"
            and "realtime.worker" in target_info.get("url", "")
        )

    def handle_worker_cdp_message(event):
        session_id = event.get(
            "sessionId",
            ""
        )

        if session_id not in worker_sessions:
            return

        try:
            message = json.loads(
                event.get(
                    "message",
                    "{}"
                )
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return

        if message.get("method") != "Network.webSocketFrameReceived":
            return

        payload_text = decode_frame_payload(
            message.get(
                "params",
                {}
            ).get(
                "response",
                {}
            ).get(
                "payloadData",
                ""
            )
        )

        if "participants" not in payload_text.lower():
            return

        rows_written = 0

        for payload in json_payloads_from_text(
            payload_text
        ):
            rows_written += process_worker_participant_payload(
                payload,
                page_source(
                    page.url
                ),
                observed_context_hashes
            )

        if not rows_written:
            return

        worker_stats["rows"] += rows_written
        current_time = time.monotonic()

        if current_time - worker_stats["last_log"] >= 10:
            log(
                "Captured "
                f"{worker_stats['rows']} worker participant context rows."
            )
            worker_stats["rows"] = 0
            worker_stats["last_log"] = current_time

    page.on(
        "response",
        lambda response: asyncio.create_task(
            handle_response(
                response
            )
        )
    )

    try:
        cdp_session = await page.context.new_cdp_session(
            page
        )

        await cdp_session.send(
            "Network.enable"
        )

        async def send_worker_cdp(session_id, method, params=None):
            message_id = worker_message_counter["next"]
            worker_message_counter["next"] += 1
            await cdp_session.send(
                "Target.sendMessageToTarget",
                {
                    "sessionId": session_id,
                    "message": json.dumps(
                        {
                            "id": message_id,
                            "method": method,
                            "params": params or {},
                        }
                    ),
                }
            )

        async def attach_realtime_worker_target(target_id):
            if not target_id or target_id in attached_worker_targets:
                return

            try:
                attached = await cdp_session.send(
                    "Target.attachToTarget",
                    {
                        "targetId": target_id,
                        "flatten": False,
                    }
                )
                session_id = attached.get(
                    "sessionId",
                    ""
                )

                if not session_id:
                    return

                attached_worker_targets.add(
                    target_id
                )
                worker_sessions.add(
                    session_id
                )

                await send_worker_cdp(
                    session_id,
                    "Network.enable"
                )

                log(
                    "Installed realtime worker participant observer."
                )

            except Exception as exc:
                log(
                    f"WARNING: Could not attach realtime worker observer: {exc}"
                )

        cdp_session.on(
            "Network.webSocketCreated",
            lambda event: websocket_urls.update(
                {
                    event.get(
                        "requestId",
                        ""
                    ): event.get(
                        "url",
                        ""
                    )
                }
            )
        )
        cdp_session.on(
            "Network.webSocketFrameReceived",
            handle_websocket_frame
        )
        cdp_session.on(
            "Target.receivedMessageFromTarget",
            handle_worker_cdp_message
        )
        cdp_session.on(
            "Target.targetCreated",
            lambda event: (
                asyncio.create_task(
                    attach_realtime_worker_target(
                        event.get(
                            "targetInfo",
                            {}
                        ).get(
                            "targetId",
                            ""
                        )
                    )
                )
                if is_realtime_worker_target(
                    event.get(
                        "targetInfo",
                        {}
                    )
                )
                else None
            )
        )

        try:
            targets = await cdp_session.send(
                "Target.getTargets"
            )

            for target_info in targets.get(
                "targetInfos",
                []
            ):
                if is_realtime_worker_target(
                    target_info
                ):
                    await attach_realtime_worker_target(
                        target_info.get(
                            "targetId",
                            ""
                        )
                    )

            await cdp_session.send(
                "Target.setDiscoverTargets",
                {
                    "discover": True,
                }
            )

        except Exception as exc:
            log(
                f"WARNING: Could not scan realtime workers: {exc}"
            )

        setattr(
            page,
            "_aviator_context_cdp_session",
            cdp_session
        )

        log(
            "Installed WebSocket context observer through Chrome DevTools."
        )

    except Exception as exc:
        log(
            f"WARNING: Could not install WebSocket context observer: {exc}"
        )

    log(
        "Installed game round ID/context observer for bet and round responses."
    )


async def read_visible_provably_fair_seed(page):
    try:
        return await page.evaluate(
            """
            () => {
              const text = document.body.innerText || "";

              if (!/PROVABLY FAIR/i.test(text)) {
                return null;
              }

              const values = Array.from(
                document.querySelectorAll("input, textarea")
              )
                .map((node) => String(node.value || "").trim())
                .filter((value) => /^[a-f0-9]{12,128}$/i.test(value));

              if (values.length < 2) {
                return null;
              }

              values.sort((left, right) => left.length - right.length);

              return {
                nextSeed: values[0] || "",
                serverNextHash: values[values.length - 1] || "",
              };
            }
            """
        )
    except Exception:
        return None


# =========================================================
# FIND NEW ROUNDS
# =========================================================

def find_new_values(old_snapshot, new_snapshot):
    """
    Assumes page history is newest -> oldest.

    Example:

    OLD:
    [3.20, 1.50, 8.10, 2.00]

    NEW:
    [4.30, 3.20, 1.50, 8.10]

    Returns:
    [4.30]

    This is more reliable than simply testing
    whether newest != last_seen.
    """

    if not old_snapshot:

        return []

    if not new_snapshot:

        return []

    # Try to determine how many values were inserted
    # at the beginning of the history list.
    #
    # The page can occasionally revise or lazy-load older history entries.
    # Because of that, requiring the entire overlapping tail to match is too
    # strict and can miss valid new rounds. Anchor on the newest stable prefix
    # of the previous snapshot instead.

    max_shift = min(
        len(new_snapshot),
        200
    )

    anchor_lengths = [
        80,
        50,
        30,
        20,
        12,
        8,
        5,
        3
    ]

    for anchor_length in anchor_lengths:

        if len(old_snapshot) < anchor_length:
            continue

        old_anchor = old_snapshot[
            :anchor_length
        ]

        for shift in range(
            1,
            max_shift + 1
        ):

            if shift + anchor_length > len(new_snapshot):
                break

            new_anchor = new_snapshot[
                shift:shift + anchor_length
            ]

            if old_anchor == new_anchor:

                return new_snapshot[:shift]

    # No shift found.
    #
    # This can happen after:
    # - refresh
    # - connection loss
    # - selector/order changes
    #
    # Don't blindly import everything.
    return []


def recover_new_values_from_recent_csv(new_snapshot, recent_values):
    if not new_snapshot or not recent_values:
        return []

    max_shift = min(
        len(new_snapshot),
        40
    )
    anchor_lengths = [
        12,
        8,
        5,
        3,
    ]

    for anchor_length in anchor_lengths:
        if len(recent_values) < anchor_length:
            continue

        anchor = list(
            reversed(
                recent_values[-anchor_length:]
            )
        )

        for shift in range(
            0,
            max_shift + 1
        ):
            if shift + anchor_length > len(new_snapshot):
                break

            if new_snapshot[shift:shift + anchor_length] == anchor:
                return new_snapshot[:shift]

    return []


# =========================================================
# SELECT AVIATRIX TAB
# =========================================================

def page_matches_preferred_url(page_url, preferred_url):
    if not preferred_url:
        return False

    try:
        preferred = urlparse(
            preferred_url
        )
        current = urlparse(
            page_url
        )
    except ValueError:
        return False

    if not preferred.netloc:
        return False

    return current.netloc.lower().endswith(
        preferred.netloc.lower()
    )


def page_is_real_aviatrix(page_url):
    lowered = page_url.lower()

    return (
        "game.aviatrix.bet" in lowered
        and "isdemo=false" in lowered
    )


def page_source(page_url):
    lowered = page_url.lower()

    if "isdemo=false" in lowered:
        return "real"

    if "isdemo=true" in lowered or "demo.aviatrix.bet" in lowered:
        return "demo"

    return "unknown"


def source_matches_required(current_source, required_source):
    if not required_source:
        return True

    normalized = str(required_source).strip().lower()

    if normalized in ("game", "any", "all"):
        return current_source in ("real", "demo", "unknown")

    if normalized in ("live", "real_or_demo", "demo_or_real"):
        return current_source in ("real", "demo")

    return current_source == normalized


async def find_aviatrix_page(browser, preferred_url=None, required_source=None):

    if not browser.contexts:
        return None

    pages = [
        page
        for context in browser.contexts
        for page in context.pages
    ]

    if not pages:
        return None

    # Prefer the configured real game page so an open demo tab does not win.

    for page in pages:

        try:

            if page_matches_preferred_url(
                page.url,
                preferred_url
            ):
                if not source_matches_required(
                    page_source(page.url),
                    required_source
                ):
                    continue

                return page

        except Exception:

            continue

    # Prefer a logged-in real Aviatrix game tab over demo.

    for page in pages:

        try:

            if page_is_real_aviatrix(
                page.url
            ):
                if not source_matches_required(
                    page_source(page.url),
                    required_source
                ):
                    continue

                return page

        except Exception:

            continue

    # Fall back to the Aviatrix game tab.

    for page in pages:

        try:

            url = page.url.lower()

            title = (
                await page.title()
            ).lower()

            if (
                "aviatrix" in title
                or
                "game.aviatrix.bet" in url
                or
                "demo.aviatrix.bet" in url
            ):
                if not source_matches_required(
                    page_source(page.url),
                    required_source
                ):
                    continue

                return page

        except Exception:

            continue

    return None


# =========================================================
# CONNECT
# =========================================================

async def connect_to_chrome(playwright):

    browser = await playwright.chromium.connect_over_cdp(
        "http://127.0.0.1:9222"
    )

    return browser


# =========================================================
# INITIAL HISTORY
# =========================================================

async def initialize_history(
    page,
    selector,
    source
):

    state = load_state()

    saved_snapshot = state.get(
        "snapshot",
        []
    )

    current = await page_read_with_timeout(
        read_multipliers_reliably(
            page,
            selector
        ),
        PAGE_READ_TIMEOUT_SECONDS * 3,
        default=[]
    )

    if not current:

        if saved_snapshot:

            log(
                "No visible multipliers found during startup; using saved snapshot until history returns."
            )

            return saved_snapshot

        log(
            "No visible multipliers found during startup."
        )

        return []

    log(
        f"Found {len(current)} visible multipliers."
    )

    # First-ever run:
    #
    # Save currently visible history oldest -> newest.

    if not saved_snapshot:

        log(
            "No previous collector state found."
        )

        log(
            "Saving initial visible history."
        )

        for multiplier in reversed(
            current
        ):

            append_round(
                multiplier,
                source=source
            )

        log(
            f"Saved {len(current)} initial multipliers."
        )

    else:

        # Collector was previously running.
        # Try to determine whether any rounds happened
        # between shutdown and restart.

        new_values = find_new_values(
            saved_snapshot,
            current
        )

        if new_values:

            if len(new_values) > MAX_STARTUP_RECOVERY_VALUES:

                log(
                    "WARNING: Startup recovery found "
                    f"{len(new_values)} possible new rounds, which is above "
                    f"the safe limit of {MAX_STARTUP_RECOVERY_VALUES}. "
                    "Resetting snapshot without appending old history."
                )

            else:

                log(
                    f"Detected {len(new_values)} rounds since previous snapshot."
                )

                for multiplier in reversed(
                    new_values
                ):

                    append_round(
                        multiplier,
                        source=source
                    )

        else:

            log(
                "Existing collector state loaded."
            )

    save_state(
        current
    )

    return current


# =========================================================
# MONITOR
# =========================================================

async def monitor_page(
    page,
    selector,
    poll_seconds,
    heartbeat_seconds,
    snapshot_scan_seconds,
    minimum_new_round_gap_seconds,
    required_source=None,
    collect_round_context=True,
    participant_count_selector=DEFAULT_PARTICIPANT_COUNT_SELECTOR,
    auto_recover_no_visible=True,
    no_visible_recovery_seconds=DEFAULT_NO_VISIBLE_RECOVERY_SECONDS,
    no_visible_recovery_cooldown_seconds=DEFAULT_NO_VISIBLE_RECOVERY_COOLDOWN_SECONDS,
    page_reload_timeout_seconds=DEFAULT_PAGE_RELOAD_TIMEOUT_SECONDS,
    page_reload_settle_seconds=DEFAULT_PAGE_RELOAD_SETTLE_SECONDS
):

    current_source = page_source(
        page.url
    )

    if not source_matches_required(
        current_source,
        required_source
    ):

        raise RuntimeError(
            "Required round source is "
            f"{required_source}, but current page source is {current_source}."
        )

    previous_snapshot = await initialize_history(
        page,
        selector,
        current_source
    )

    log(
        f"Round source: {current_source}"
    )

    await install_round_id_observer(
        page,
        collect_round_context
    )

    await install_history_watcher(
        page,
        selector,
        previous_snapshot
    )

    await install_game_status_watcher(
        page
    )

    last_heartbeat = time.time()

    last_no_visible_log = 0

    no_visible_since = None

    last_no_visible_recovery_at = 0

    no_overlap_count = 0

    last_snapshot_scan = 0

    last_seed_scan = 0

    last_participant_scan = 0

    last_participant_count = None

    last_participant_write_at = 0

    last_participant_log_at = 0

    last_participant_table_scan = 0

    last_participant_table_signature = None

    last_participant_table_write_at = 0

    last_participant_table_log_at = 0

    persisted_state = load_state()

    game_status = (
        persisted_state.get(
            "game_status",
            {}
        )
        if isinstance(
            persisted_state,
            dict
        )
        else {}
    )

    if not isinstance(
        game_status,
        dict
    ):
        game_status = {}

    game_status_event_id = int(
        game_status.get(
            "event_id",
            0
        )
        or 0
    )

    last_game_status_scan = 0

    last_game_status_write_at = 0

    last_game_status_signature = None

    last_handled_finish_event_id = game_status.get(
        "finish_event_id"
    )

    last_finish_force_at = 0

    observed_seed_pairs = load_observed_seed_pairs()

    last_appended_multiplier = None

    last_appended_at = 0

    fast_poll_seconds = min(
        poll_seconds,
        max(
            PAGE_WATCHER_INTERVAL_MS / 1000,
            0.05
        )
    )

    log(
        "Live monitoring started "
        f"(watcher drain every {fast_poll_seconds:.2f}s, "
        f"snapshot scan every {snapshot_scan_seconds:.2f}s)."
    )

    def update_game_status(update):
        nonlocal game_status
        nonlocal game_status_event_id

        if not update:
            return

        previous_phase = game_status.get(
            "phase"
        )
        previous_round_state = game_status.get(
            "round_state"
        )
        next_status = dict(
            game_status
        )
        next_status.update(
            update
        )

        if (
            update.get("source") == "live_score_dom"
            and update.get("phase") == "running"
        ):
            next_status["round_state"] = "STATE_RUN"
            next_status["is_preparing"] = False

        if "liveMultiplier" in next_status:
            next_status["live_multiplier"] = next_status.pop(
                "liveMultiplier"
            )

        next_status["observed_at"] = now_string()
        next_status["game_source"] = current_source

        phase_changed = (
            next_status.get(
                "phase"
            )
            != previous_phase
        )
        round_state_changed = (
            next_status.get(
                "round_state"
            )
            and next_status.get(
                "round_state"
            )
            != previous_round_state
        )

        if phase_changed or round_state_changed:
            game_status_event_id += 1
            next_status["event_id"] = game_status_event_id
            next_status["phase_started_at"] = next_status["observed_at"]

        if next_status.get("phase") == "finished":
            next_status["last_finish_at"] = next_status["observed_at"]
            next_status["finish_event_id"] = game_status_event_id
            next_status["live_multiplier"] = None

        if next_status.get("phase") == "running":
            next_status["last_run_at"] = next_status["observed_at"]

        game_status = next_status

    def game_status_signature():
        try:
            return json.dumps(
                game_status,
                sort_keys=True
            )
        except TypeError:
            return str(
                game_status
            )

    def persist_game_status_if_needed(force=False):
        nonlocal last_game_status_signature
        nonlocal last_game_status_write_at

        if not game_status:
            return

        current_time = time.monotonic()
        signature = game_status_signature()
        changed = signature != last_game_status_signature
        heartbeat_due = (
            current_time - last_game_status_write_at
            >= GAME_STATUS_HEARTBEAT_SECONDS
        )
        minimum_gap_passed = (
            current_time - last_game_status_write_at
            >= GAME_STATUS_MIN_WRITE_SECONDS
        )

        if not force and not (
            heartbeat_due
            or (
                changed
                and minimum_gap_passed
            )
        ):
            return

        save_state(
            previous_snapshot,
            game_status
        )
        last_game_status_signature = signature
        last_game_status_write_at = current_time

    async def handle_console_message(message):
        message_text = getattr(
            message,
            "text",
            ""
        )

        if callable(
            message_text
        ):
            try:
                message_text = message_text()
            except Exception:
                message_text = ""

        update_game_status(
            parse_console_game_status_text(
                message_text
            )
        )

        for arg in getattr(
            message,
            "args",
            []
        ) or []:
            try:
                value = await arg.json_value()
            except Exception:
                continue

            update_game_status(
                parse_console_game_status_value(
                    value
                )
            )

    page.on(
        "console",
        lambda message: asyncio.create_task(
            handle_console_message(
                message
            )
        )
    )

    def append_live_round(multiplier, timestamp=None, round_id=None):
        nonlocal last_appended_multiplier
        nonlocal last_appended_at

        current_time = time.monotonic()

        if (
            last_appended_multiplier == multiplier
            and current_time - last_appended_at < minimum_new_round_gap_seconds
        ):
            log(
                f"Skipped duplicate live round: {multiplier:.2f}x"
            )
            return

        append_round(
            multiplier,
            timestamp,
            round_id,
            current_source
        )

        last_appended_multiplier = multiplier
        last_appended_at = current_time

        log(
            f"NEW ROUND: {multiplier:.2f}x"
        )

    async def recover_no_visible_multipliers():
        nonlocal previous_snapshot
        nonlocal no_overlap_count
        nonlocal no_visible_since
        nonlocal last_no_visible_recovery_at
        nonlocal last_snapshot_scan
        nonlocal last_seed_scan
        nonlocal last_game_status_scan
        nonlocal current_source

        current_time = time.monotonic()
        seconds_hidden = (
            current_time - no_visible_since
            if no_visible_since is not None
            else 0
        )

        last_no_visible_recovery_at = current_time

        log(
            "WARNING: Multiplier history has been invisible for "
            f"{seconds_hidden:.1f}s. Auto-reloading the Aviatrix tab."
        )

        await page_read_with_timeout(
            page.reload(
                wait_until="domcontentloaded",
                timeout=page_reload_timeout_seconds * 1000
            ),
            page_reload_timeout_seconds + 2,
            default=None
        )

        await asyncio.sleep(
            page_reload_settle_seconds
        )

        live_source = page_source(
            page.url
        )

        if not source_matches_required(
            live_source,
            required_source
        ):
            raise RuntimeError(
                "Required round source changed during auto-recovery."
            )

        current_source = live_source

        recovered_snapshot = await page_read_with_timeout(
            read_multipliers_reliably(
                page,
                selector,
                attempts=8,
                delay_seconds=0.5
            ),
            max(
                page_reload_timeout_seconds,
                PAGE_READ_TIMEOUT_SECONDS * 8
            ),
            default=[]
        )

        if not recovered_snapshot:
            log(
                "WARNING: Auto-reload did not restore visible multipliers. "
                "Reconnecting collector."
            )

            raise RuntimeError(
                "Auto-recovery could not restore visible multipliers."
            )

        recovered_values = []

        if previous_snapshot:
            recovered_values = find_new_values(
                previous_snapshot,
                recovered_snapshot
            )

        if not recovered_values:
            recovered_values = recover_new_values_from_recent_csv(
                recovered_snapshot,
                load_recent_round_values()
            )

        if recovered_values:
            if len(recovered_values) > MAX_NEW_VALUES_PER_SCAN:
                log(
                    "WARNING: Auto-recovery found "
                    f"{len(recovered_values)} possible new rounds, above safe "
                    f"limit of {MAX_NEW_VALUES_PER_SCAN}. Resetting snapshot "
                    "without appending old history."
                )
            else:
                log(
                    "Auto-recovery recovered "
                    f"{len(recovered_values)} missed rounds from visible history."
                )

                for multiplier in reversed(
                    recovered_values
                ):
                    append_live_round(
                        multiplier
                    )

        previous_snapshot = recovered_snapshot

        save_state(
            previous_snapshot,
            game_status
        )

        await install_history_watcher(
            page,
            selector,
            previous_snapshot
        )

        await install_game_status_watcher(
            page
        )

        no_overlap_count = 0
        no_visible_since = None
        last_snapshot_scan = 0
        last_seed_scan = 0
        last_game_status_scan = 0

        log(
            "Auto-recovery restored multiplier visibility "
            f"({len(previous_snapshot)} visible multipliers)."
        )

    async def flush_history_watcher_queue():
        nonlocal previous_snapshot
        nonlocal no_overlap_count

        watcher_state = await page_read_with_timeout(
            drain_history_watcher(
                page
            ),
            WATCHER_DRAIN_TIMEOUT_SECONDS,
            default=None
        )

        if not watcher_state or not watcher_state.get("queue"):
            return False

        queued_events = watcher_state.get(
            "queue",
            []
        )

        if len(queued_events) > MAX_NEW_VALUES_PER_SCAN:

            log(
                "WARNING: Page watcher found "
                f"{len(queued_events)} possible rounds, above safe limit. "
                "Dropping queued batch and resetting snapshot."
            )

        else:

            for event in queued_events:

                try:
                    if isinstance(event, dict):
                        multiplier = float(
                            event.get(
                                "value"
                            )
                        )
                        round_id = event.get(
                            "roundId"
                        )
                        detected_at = timestamp_from_millis(
                            event.get(
                                "detectedAt"
                            )
                        )
                    else:
                        multiplier = float(
                            event
                        )
                        round_id = None
                        detected_at = now_string()

                    if multiplier < 1:
                        continue

                except (TypeError, ValueError):
                    continue

                append_live_round(
                    multiplier,
                    detected_at,
                    round_id
                )

        previous_snapshot = watcher_state.get(
            "snapshot",
            previous_snapshot
        )

        save_state(
            previous_snapshot,
            game_status
        )

        no_overlap_count = 0

        return True

    while True:

        try:

            live_source = page_source(
                page.url
            )

            if not source_matches_required(
                live_source,
                required_source
            ):

                log(
                    "WARNING: Required round source changed from "
                    f"{required_source} to {live_source}. Reconnecting without writing."
                )

                raise RuntimeError(
                    "Required round source changed."
                )

            current_source = live_source

            if time.time() - last_game_status_scan >= GAME_STATUS_SCAN_SECONDS:
                last_game_status_scan = time.time()

                watcher_status = await page_read_with_timeout(
                    read_game_status_watcher(
                        page
                    ),
                    PAGE_READ_TIMEOUT_SECONDS,
                    default=None
                )

                if watcher_status:
                    update_game_status(
                        watcher_status
                    )

                persist_game_status_if_needed()

            finish_event_id = game_status.get(
                "finish_event_id"
            )

            if (
                finish_event_id
                and finish_event_id != last_handled_finish_event_id
            ):
                last_handled_finish_event_id = finish_event_id
                current_time = time.monotonic()

                if current_time - last_finish_force_at >= 1:
                    last_finish_force_at = current_time
                    last_snapshot_scan = 0

                    log(
                        "Detected finished round state; forcing immediate history read."
                    )

            await flush_history_watcher_queue()

            if (
                collect_round_context
                and time.time() - last_participant_scan >= PARTICIPANT_SCAN_SECONDS
            ):
                last_participant_scan = time.time()

                participant_count = await page_read_with_timeout(
                    read_participant_count(
                        page,
                        participant_count_selector
                    ),
                    PAGE_READ_TIMEOUT_SECONDS,
                    default=None
                )

                if participant_count is not None:
                    current_time = time.monotonic()
                    count_changed = participant_count != last_participant_count
                    minimum_write_gap_passed = (
                        current_time - last_participant_write_at >= PARTICIPANT_MIN_WRITE_SECONDS
                    )
                    heartbeat_due = (
                        current_time - last_participant_write_at >= PARTICIPANT_HEARTBEAT_SECONDS
                    )
                    should_write_participant_context = (
                        (
                            count_changed
                            and minimum_write_gap_passed
                        )
                        or heartbeat_due
                    )

                    if should_write_participant_context:
                        append_participant_count_context(
                            participant_count,
                            current_source
                        )
                        last_participant_write_at = current_time

                        if (
                            count_changed
                            and current_time - last_participant_log_at >= 10
                        ):
                            log(
                                "Captured live participant count from flight radar: "
                                f"{participant_count}"
                            )
                            last_participant_log_at = current_time

                        last_participant_count = participant_count

            if (
                collect_round_context
                and time.time() - last_participant_table_scan >= PARTICIPANT_TABLE_SCAN_SECONDS
            ):
                last_participant_table_scan = time.time()

                table_context = await page_read_with_timeout(
                    read_visible_participants_table(
                        page
                    ),
                    SLOW_CONTEXT_READ_TIMEOUT_SECONDS,
                    default=None
                )

                if table_context:
                    current_time = time.monotonic()
                    table_signature = visible_participants_signature(
                        table_context,
                        current_source
                    )
                    table_changed = table_signature != last_participant_table_signature
                    table_heartbeat_due = (
                        current_time - last_participant_table_write_at
                        >= PARTICIPANT_TABLE_HEARTBEAT_SECONDS
                    )

                    if table_changed or table_heartbeat_due:
                        append_visible_participants_context(
                            table_context,
                            current_source
                        )
                        last_participant_table_signature = table_signature
                        last_participant_table_write_at = current_time

                        if (
                            table_changed
                            and current_time - last_participant_table_log_at >= 5
                        ):
                            log(
                                "Captured visible participants table aggregate: "
                                f"{int(table_context.get('visible_rows', 0))} rows, "
                                f"{context_number(table_context.get('total_bet')) or 'unknown'} total bet."
                            )
                            last_participant_table_log_at = current_time

            await flush_history_watcher_queue()

            did_snapshot_scan = False
            current_snapshot = []

            if time.time() - last_snapshot_scan >= snapshot_scan_seconds:

                did_snapshot_scan = True

                last_snapshot_scan = time.time()

                current_snapshot = await page_read_with_timeout(
                    read_multipliers(
                        page,
                        selector
                    ),
                    PAGE_READ_TIMEOUT_SECONDS,
                    default=[]
                )

            if time.time() - last_seed_scan >= 5:

                last_seed_scan = time.time()

                seed_data = await page_read_with_timeout(
                    read_visible_provably_fair_seed(
                        page
                    ),
                    PAGE_READ_TIMEOUT_SECONDS,
                    default=None
                )

                if seed_data:

                    next_seed = seed_data.get(
                        "nextSeed",
                        ""
                    )
                    server_next_hash = seed_data.get(
                        "serverNextHash",
                        ""
                    )
                    seed_pair = (
                        next_seed,
                        server_next_hash
                    )

                    if seed_pair not in observed_seed_pairs:

                        observed_seed_pairs.add(
                            seed_pair
                        )

                        append_provably_fair_seed(
                            next_seed,
                            server_next_hash,
                            "visible_provably_fair"
                        )

                        log(
                            "Captured visible provably fair seed/hash."
                        )

            if current_snapshot:
                no_visible_since = None

                if previous_snapshot:

                    new_values = find_new_values(
                        previous_snapshot,
                        current_snapshot
                    )

                    if new_values:

                        if len(new_values) > MAX_NEW_VALUES_PER_SCAN:

                            log(
                                "WARNING: Found "
                                f"{len(new_values)} possible new rounds in one scan, "
                                f"above safe limit of {MAX_NEW_VALUES_PER_SCAN}. "
                                "Resetting snapshot without appending old history."
                            )

                        else:

                            # new_values are newest -> older.
                            #
                            # Write chronological order.

                            for multiplier in reversed(
                                new_values
                            ):

                                append_live_round(
                                    multiplier
                                )

                        no_overlap_count = 0

                    elif current_snapshot != previous_snapshot:

                        recovered_values = recover_new_values_from_recent_csv(
                            current_snapshot,
                            load_recent_round_values()
                        )

                        if recovered_values:

                            if len(recovered_values) > MAX_NEW_VALUES_PER_SCAN:

                                log(
                                    "WARNING: CSV anchor recovery found "
                                    f"{len(recovered_values)} possible new rounds, "
                                    f"above safe limit of {MAX_NEW_VALUES_PER_SCAN}. "
                                    "Resetting snapshot without appending old history."
                                )

                            else:

                                log(
                                    "Recovered "
                                    f"{len(recovered_values)} rounds using recent CSV anchor."
                                )

                                for multiplier in reversed(
                                    recovered_values
                                ):

                                    append_live_round(
                                        multiplier
                                    )

                            no_overlap_count = 0

                        else:

                            no_overlap_count += 1

                            log(
                                "WARNING: Snapshot changed but no reliable overlap was found "
                                f"({no_overlap_count}/3)."
                            )

                            if no_overlap_count < 3:

                                await asyncio.sleep(
                                    fast_poll_seconds
                                )

                                continue

                            log(
                                "Resetting snapshot to current visible history so live tracking can continue."
                            )

                            no_overlap_count = 0

                    else:

                        no_overlap_count = 0

                previous_snapshot = current_snapshot

                save_state(
                    current_snapshot,
                    game_status
                )

            elif did_snapshot_scan:
                current_time = time.monotonic()

                if no_visible_since is None:
                    no_visible_since = current_time

                no_visible_elapsed = current_time - no_visible_since

                if time.time() - last_no_visible_log >= 5:

                    log(
                        "WARNING: No multipliers currently visible."
                    )

                    last_no_visible_log = time.time()

                if (
                    auto_recover_no_visible
                    and no_visible_elapsed >= no_visible_recovery_seconds
                    and current_time - last_no_visible_recovery_at
                    >= no_visible_recovery_cooldown_seconds
                ):
                    await recover_no_visible_multipliers()

            # Heartbeat

            if (
                time.time()
                - last_heartbeat
                >= heartbeat_seconds
            ):

                log(
                    "Collector heartbeat: running normally."
                )

                last_heartbeat = time.time()

            await asyncio.sleep(
                fast_poll_seconds
            )

        except Exception as exc:

            log(
                f"Page read error: {exc}"
            )

            raise


# =========================================================
# MAIN
# =========================================================

async def main():

    cfg = load_config()

    ensure_files()

    lock_file = acquire_collector_lock()

    # Keep the lock file handle alive for the whole collector process.
    _ = lock_file

    backup_rounds_csv()

    selector = cfg.get(
        "history_selectors",
        cfg.get(
            "history_selector",
            ".text-w-60"
        )
    )

    poll_seconds = float(
        cfg.get(
            "poll_seconds",
            1.0
        )
    )

    heartbeat_seconds = float(
        cfg.get(
            "heartbeat_seconds",
            60
        )
    )

    snapshot_scan_seconds = float(
        cfg.get(
            "snapshot_scan_seconds",
            DEFAULT_SNAPSHOT_SCAN_SECONDS
        )
    )

    reconnect_seconds = float(
        cfg.get(
            "reconnect_seconds",
            5
        )
    )

    minimum_new_round_gap_seconds = float(
        cfg.get(
            "minimum_new_round_gap_seconds",
            1.5
        )
    )
    required_source = str(
        cfg.get(
            "require_source",
            ""
        )
    ).strip().lower() or None

    collect_round_context = bool(
        cfg.get(
            "collect_round_context",
            True
        )
    )

    participant_count_selector = cfg.get(
        "participant_count_selector",
        DEFAULT_PARTICIPANT_COUNT_SELECTOR
    )

    auto_recover_no_visible = bool(
        cfg.get(
            "auto_recover_no_visible",
            True
        )
    )

    no_visible_recovery_seconds = float(
        cfg.get(
            "no_visible_recovery_seconds",
            DEFAULT_NO_VISIBLE_RECOVERY_SECONDS
        )
    )

    no_visible_recovery_cooldown_seconds = float(
        cfg.get(
            "no_visible_recovery_cooldown_seconds",
            DEFAULT_NO_VISIBLE_RECOVERY_COOLDOWN_SECONDS
        )
    )

    page_reload_timeout_seconds = float(
        cfg.get(
            "page_reload_timeout_seconds",
            DEFAULT_PAGE_RELOAD_TIMEOUT_SECONDS
        )
    )

    page_reload_settle_seconds = float(
        cfg.get(
            "page_reload_settle_seconds",
            DEFAULT_PAGE_RELOAD_SETTLE_SECONDS
        )
    )

    log(
        "=================================="
    )

    log(
        "Aviatrix collector starting"
    )

    log(
        f"Selector: {selector}"
    )

    log(
        "Round context aggregates: "
        f"{'enabled' if collect_round_context else 'disabled'}"
    )

    log(
        f"Participant count selector: {participant_count_selector}"
    )

    log(
        "Auto recovery: "
        f"{'enabled' if auto_recover_no_visible else 'disabled'} "
        f"(no visible threshold {no_visible_recovery_seconds:.1f}s, "
        f"cooldown {no_visible_recovery_cooldown_seconds:.1f}s)."
    )

    async with async_playwright() as p:

        while True:

            browser = None

            try:

                log(
                    "Connecting to Chrome..."
                )

                browser = await connect_to_chrome(
                    p
                )

                log(
                    "Connected to Chrome."
                )

                page = await find_aviatrix_page(
                    browser,
                    cfg.get(
                        "game_url"
                    ),
                    required_source
                )

                if page is None:

                    log(
                        "Required Aviatrix tab not found."
                    )

                    source_hint = (
                        "Open any Aviatrix game tab, demo or real, in the debug Chrome window."
                        if required_source in ("game", "any", "all", "live", "real_or_demo", "demo_or_real")
                        else "Open the main real Aviatrix game tab (game.aviatrix.bet with isDemo=false) in the debug Chrome window."
                    )

                    log(
                        source_hint
                    )

                    await asyncio.sleep(
                        reconnect_seconds
                    )

                    continue

                title = await page.title()

                log(
                    f"Using page: {title}"
                )

                log(
                    f"URL: {page.url}"
                )

                await monitor_page(
                    page,
                    selector,
                    poll_seconds,
                    heartbeat_seconds,
                    snapshot_scan_seconds,
                    minimum_new_round_gap_seconds,
                    required_source,
                    collect_round_context,
                    participant_count_selector,
                    auto_recover_no_visible,
                    no_visible_recovery_seconds,
                    no_visible_recovery_cooldown_seconds,
                    page_reload_timeout_seconds,
                    page_reload_settle_seconds
                )

            except KeyboardInterrupt:

                log(
                    "Collector stopped by user."
                )

                return

            except Exception as exc:

                log(
                    f"Collector error: {exc}"
                )

                log(
                    f"Retrying in {reconnect_seconds} seconds..."
                )

                await asyncio.sleep(
                    reconnect_seconds
                )


if __name__ == "__main__":
    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        print(
            "\nCollector stopped."
        )
