const state = {
  timer: null,
  lastRoundCount: 0,
  inFlight: false,
  pendingRefresh: false,
  lastRenderSignature: "",
};

const REFRESH_MS = 100;
const LIVE_ENDPOINT = "/api/live";
const PARTICIPANT_CONTEXT_FRESH_SECONDS = 5;

const elements = {
  status: document.querySelector(".status"),
  statusText: document.getElementById("statusText"),
  fastPredictionText: document.getElementById("fastPredictionText"),
  fastPredictionMeta: document.getElementById("fastPredictionMeta"),
  signalStrengthText: document.getElementById("signalStrengthText"),
  bigWatchText: document.getElementById("bigWatchText"),
  fastSignalText: document.getElementById("fastSignalText"),
  cashoutGuideText: document.getElementById("cashoutGuideText"),
  fastMain: document.querySelector(".fast-main"),
  previousPredictionStatus: document.getElementById("previousPredictionStatus"),
  accuracySummaryText: document.getElementById("accuracySummaryText"),
  sourceModeText: document.getElementById("sourceModeText"),
  roundContextText: document.getElementById("roundContextText"),
  latestMultiplier: document.getElementById("latestMultiplier"),
  roundCount: document.getElementById("roundCount"),
  medianValue: document.getElementById("medianValue"),
  p90Value: document.getElementById("p90Value"),
  maxValue: document.getElementById("maxValue"),
  lastUpdated: document.getElementById("lastUpdated"),
  latestPattern: document.getElementById("latestPattern"),
  matchCount: document.getElementById("matchCount"),
  predictionList: document.getElementById("predictionList"),
  overallList: document.getElementById("overallList"),
  backtestList: document.getElementById("backtestList"),
  trackingSummary: document.getElementById("trackingSummary"),
  trackingList: document.getElementById("trackingList"),
  recentRounds: document.getElementById("recentRounds"),
  riskNote: document.getElementById("riskNote"),
  chart: document.getElementById("roundChart"),
  lookbackSelect: document.getElementById("lookbackSelect"),
  minMatchesSelect: document.getElementById("minMatchesSelect"),
  refreshButton: document.getElementById("refreshButton"),
};

function formatMultiplier(value) {
  if (value === null || value === undefined) {
    return "--";
  }

  return `${Number(value).toFixed(2)}x`;
}

function formatCashoutMultiplier(value) {
  if (value === null || value === undefined) {
    return "--";
  }

  const multiplier = Number(value);

  if (!Number.isFinite(multiplier)) {
    return "--";
  }

  return `${multiplier.toFixed(2)}x`;
}

function formatPercent(value) {
  if (value === null || value === undefined) {
    return "--";
  }

  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatSignedPercent(value) {
  if (value === null || value === undefined) {
    return "--";
  }

  const number = Number(value);
  const sign = number > 0 ? "+" : "";
  return `${sign}${(number * 100).toFixed(1)}%`;
}

function formatContextNumber(value) {
  if (value === null || value === undefined) {
    return "--";
  }

  return Number(value).toFixed(2);
}

function formatMoney(value) {
  if (value === null || value === undefined) {
    return "--";
  }

  const amount = Number(value);

  if (!Number.isFinite(amount)) {
    return "--";
  }

  if (Math.abs(amount) >= 1000000) {
    return `${(amount / 1000000).toFixed(2)}M`;
  }

  if (Math.abs(amount) >= 1000) {
    return `${(amount / 1000).toFixed(2)}K`;
  }

  return amount.toFixed(2);
}

function formatIndianMoney(value) {
  if (value === null || value === undefined) {
    return "--";
  }

  const amount = Number(value);

  if (!Number.isFinite(amount)) {
    return "--";
  }

  const absAmount = Math.abs(amount);

  if (absAmount >= 10000000) {
    return `Rs ${(amount / 10000000).toFixed(2)}Cr`;
  }

  if (absAmount >= 100000) {
    return `Rs ${(amount / 100000).toFixed(2)}L`;
  }

  if (absAmount >= 1000) {
    return `Rs ${(amount / 1000).toFixed(2)}K`;
  }

  return `Rs ${amount.toFixed(2)}`;
}

function formatDisplayMoney(value, currency) {
  const normalizedCurrency = String(currency || "").toUpperCase();

  if (normalizedCurrency === "INR") {
    return formatIndianMoney(value);
  }

  if (normalizedCurrency) {
    return `${normalizedCurrency} ${formatMoney(value)}`;
  }

  return formatMoney(value);
}

function displayMoneyValue(context, key) {
  const displayKey = `display_${key}`;

  if (context[displayKey] !== null && context[displayKey] !== undefined) {
    return context[displayKey];
  }

  return context[key];
}

function setStatus(kind, text) {
  elements.status.classList.remove("live", "error");

  if (kind) {
    elements.status.classList.add(kind);
  }

  elements.statusText.textContent = text;
}

function formatAge(seconds) {
  if (seconds === null || seconds === undefined) {
    return "unknown";
  }

  if (seconds < 60) {
    return `${seconds}s`;
  }

  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}m ${remainingSeconds}s`;
}

function liveDataLabel(data) {
  const selection = data.data_selection || {};
  const mode = selection.using_trusted_sources
    ? "REAL + HISTORY"
    : selection.using_source_only || selection.mode === "real"
    ? "REAL DATA"
    : "ALL DATA";

  if (data.ingest && data.ingest.is_stale) {
    return `${mode} STALE`;
  }

  return `${mode} LIVE`;
}

function cleanCashoutGuide(text) {
  return String(text || "waiting").replace(/^Cashout guide:\s*/i, "");
}

function liveDataDetail(data, cashoutText) {
  const ageText = data.ingest
    ? `last round ${formatAge(data.ingest.last_round_age_seconds)} ago`
    : "last round unknown";
  const selection = data.data_selection || {};
  const roundCount = selection.using_trusted_sources
    ? `${selection.trusted_rounds} real+history rounds`
    : selection.using_source_only
    ? `${selection.source_rounds} real rounds`
    : `${data.summary ? data.summary.rounds : "--"} rounds`;

  return `${ageText} - ${roundCount} - ${cleanCashoutGuide(cashoutText)}`;
}

function probabilityColor(probability) {
  if (probability >= 0.65) {
    return "var(--green)";
  }

  if (probability >= 0.35) {
    return "var(--amber)";
  }

  return "var(--red)";
}

function targetClass(value) {
  return `target-${Number(value).toFixed(2).replace(".", "-")}`;
}

function signalClass(signal) {
  return `signal-${String(signal || "neutral").toLowerCase()}`;
}

function findTargetPrediction(predictions, target) {
  return predictions.find((item) => Number(item.target).toFixed(2) === Number(target).toFixed(2));
}

function buildDirectPrediction(predictions) {
  const clearPredictions = predictions.filter((item) => item.clear_signal);
  const p100 = findTargetPrediction(clearPredictions, 100);
  const p50 = findTargetPrediction(clearPredictions, 50);
  const p25 = findTargetPrediction(clearPredictions, 25);
  const p10 = findTargetPrediction(clearPredictions, 10);
  const p15 = findTargetPrediction(predictions, 1.5);
  const p2 = findTargetPrediction(clearPredictions, 2);
  const p3 = findTargetPrediction(clearPredictions, 3);
  const p5 = findTargetPrediction(clearPredictions, 5);
  const low15 = findTargetPrediction(clearPredictions, 1.5);
  const fallback = clearPredictions
    .slice()
    .sort((left, right) => Math.abs(right.edge) - Math.abs(left.edge))[0];

  if (p100 && p100.predicted_high && p100.signal === "FAVOR") {
    return {
      prediction: p100,
      tone: "fast-high",
      text: "NEXT: MORE THAN 100.00x",
      note: "Extreme multiplier watch",
    };
  }

  if (p50 && p50.predicted_high && p50.signal === "FAVOR") {
    return {
      prediction: p50,
      tone: "fast-high",
      text: "NEXT: MORE THAN 50.00x",
      note: "Big multiplier watch",
    };
  }

  if (p25 && p25.predicted_high && p25.signal === "FAVOR") {
    return {
      prediction: p25,
      tone: "fast-high",
      text: "NEXT: MORE THAN 25.00x",
      note: "High multiplier watch",
    };
  }

  if (p10 && p10.predicted_high && p10.signal === "FAVOR") {
    return {
      prediction: p10,
      tone: "fast-high",
      text: "NEXT: MORE THAN 10.00x",
      note: "10x+ signal",
    };
  }

  if (p5 && p5.predicted_high && p5.signal === "FAVOR") {
    return {
      prediction: p5,
      tone: "fast-high",
      text: "NEXT: MORE THAN 5.00x",
      note: "Strong high-multiplier signal",
    };
  }

  if (p3 && p3.predicted_high && p3.signal === "FAVOR") {
    return {
      prediction: p3,
      tone: "fast-high",
      text: "NEXT: MORE THAN 3.00x",
      note: "Strong 3x+ signal",
    };
  }

  if (p2 && p2.predicted_high) {
    return {
      prediction: p2,
      tone: "fast-high",
      text: "NEXT: MORE THAN 2.00x",
      note: "Main 2x target is positive",
    };
  }

  if (low15 && !low15.predicted_high) {
    return {
      prediction: low15,
      tone: "fast-low",
      text: "NEXT: LESS THAN 1.50x",
      note: "Low-round risk is highest",
    };
  }

  if (low15 && low15.predicted_high && !findTargetPrediction(clearPredictions, 2)) {
    return {
      prediction: low15,
      tone: "fast-range",
      text: "NEXT: 1.50x TO 2.00x RANGE",
      note: "Above 1.5x, but 2x is weak",
    };
  }

  if (fallback) {
    return {
      prediction: fallback,
      tone: fallback.predicted_high ? "fast-high" : "fast-low",
      text: fallback.predicted_high
        ? `NEXT: MORE THAN ${Number(fallback.target).toFixed(2)}x`
        : `NEXT: LESS THAN ${Number(fallback.target).toFixed(2)}x`,
      note: "Best available target call",
    };
  }

  if (predictions.length) {
    const strongestWeak = predictions
      .slice()
      .sort((left, right) => Math.abs(right.edge) - Math.abs(left.edge))[0];

    return {
      prediction: strongestWeak,
      tone: strongestWeak.predicted_high ? "fast-range" : "fast-low",
      text: strongestWeak.predicted_high
        ? `NEXT: MORE THAN ${Number(strongestWeak.target).toFixed(2)}x`
        : `NEXT: LESS THAN ${Number(strongestWeak.target).toFixed(2)}x`,
      note: strongestWeak && strongestWeak.clear_reason
        ? strongestWeak.clear_reason
        : "Evidence is not strong enough",
      waiting: false,
      weak: true,
    };
  }

  return null;
}

function formatRangeEstimate(range) {
  if (!range) {
    return null;
  }

  const prefix = range.clear_signal ? "Predicted crash" : "Best estimate";

  if (range.maximum === null || range.maximum === undefined) {
    return `${prefix}: above ${Number(range.minimum).toFixed(2)}x`;
  }

  return `${prefix}: ${Number(range.minimum).toFixed(2)}x to ${Number(range.maximum).toFixed(2)}x`;
}

function formatMainPrediction(range, direct) {
  if (
    direct
    && !direct.weak
    && direct.prediction
    && direct.prediction.predicted_high
    && Number(direct.prediction.target) >= 2
  ) {
    return `Predicted: more than ${Number(direct.prediction.target).toFixed(2)}x`;
  }

  return formatRangeEstimate(range) || (direct ? direct.text : null);
}

function directFitsRange(direct, range) {
  if (!direct || !range || !direct.prediction) {
    return false;
  }

  const target = Number(direct.prediction.target);
  const maximum = range.maximum === null || range.maximum === undefined
    ? null
    : Number(range.maximum);

  if (direct.prediction.predicted_high) {
    return maximum === null || target <= maximum;
  }

  return maximum !== null && target >= Number(range.minimum);
}

function rangeCallText(range) {
  if (!range) {
    return "--";
  }

  if (range.maximum === null || range.maximum === undefined) {
    return `ABOVE ${Number(range.minimum).toFixed(2)}x`;
  }

  return `${Number(range.minimum).toFixed(2)}x TO ${Number(range.maximum).toFixed(2)}x`;
}

function rangeResultMatched(rangeResult, actualMultiplier) {
  if (!rangeResult || actualMultiplier === null || actualMultiplier === undefined) {
    return null;
  }

  const minimum = Number(rangeResult.minimum);
  const maximum = rangeResult.maximum === null || rangeResult.maximum === undefined
    ? null
    : Number(rangeResult.maximum);
  const actual = Number(actualMultiplier);

  if (!Number.isFinite(minimum) || !Number.isFinite(actual)) {
    return null;
  }

  return actual >= minimum && (maximum === null || actual < maximum);
}

function buildConsistentDisplay(range, direct) {
  const canUseDirect = (
    direct
    && !direct.weak
    && direct.prediction
    && direct.prediction.clear_signal
    && directFitsRange(direct, range)
  );

  if (canUseDirect) {
    return {
      prediction: direct.prediction,
      tone: direct.tone,
      predictionText: formatMainPrediction(range, direct),
      mainCallText: direct.prediction.predicted_high
        ? `MORE THAN ${Number(direct.prediction.target).toFixed(2)}x`
        : `LESS THAN ${Number(direct.prediction.target).toFixed(2)}x`,
      signalText: signalStrengthLabel(direct.prediction, direct),
      cashoutText: cashoutGuide(range, direct),
      direct,
    };
  }

  if (range) {
    return {
      prediction: null,
      tone: rangeTone(range, direct ? direct.tone : "fast-range"),
      predictionText: formatRangeEstimate(range),
      mainCallText: rangeCallText(range),
      signalText: range.clear_signal
        ? `Signal: RANGE - ${range.confidence} confidence`
        : `Signal: WEAK RANGE - ${range.clear_reason || "not scored"}`,
      cashoutText: cashoutGuide(range, null),
      direct: null,
    };
  }

  if (direct) {
    return {
      prediction: direct.prediction,
      tone: direct.tone,
      predictionText: direct.text,
      mainCallText: direct.prediction.predicted_high
        ? `MORE THAN ${Number(direct.prediction.target).toFixed(2)}x`
        : `LESS THAN ${Number(direct.prediction.target).toFixed(2)}x`,
      signalText: signalStrengthLabel(direct.prediction, direct),
      cashoutText: cashoutGuide(null, direct),
      direct,
    };
  }

  return null;
}

function formatRangeStats(range) {
  if (!range) {
    return "";
  }

  return `estimated band ${formatMultiplier(range.low)} / ${formatMultiplier(range.median)} / ${formatMultiplier(range.high)} - ${formatPercent(range.probability)} range chance`;
}

function cashoutGuide(range, direct) {
  if (
    direct
    && !direct.weak
    && direct.prediction
    && direct.prediction.predicted_high
    && Number(direct.prediction.target) >= 2
  ) {
    return `Cashout guide: before ${Number(direct.prediction.target).toFixed(2)}x`;
  }

  if (!range) {
    return "Cashout guide: waiting for range";
  }

  if (range.clear_signal === false) {
    return "Cashout guide: weak range, use low target only";
  }

  const minimum = Number(range.minimum);
  const maximum = range.maximum === null || range.maximum === undefined
    ? null
    : Number(range.maximum);
  const median = Number(range.median);

  if (maximum !== null && maximum <= 1.5) {
    return `Cashout guide: very early, before ${maximum.toFixed(2)}x`;
  }

  if (maximum !== null && maximum <= 2) {
    return `Cashout guide: before ${Math.max(1.2, minimum).toFixed(2)}x`;
  }

  if (minimum >= 5) {
    return `Cashout guide: protect before ${minimum.toFixed(2)}x`;
  }

  if (minimum >= 2) {
    return `Cashout guide: before ${minimum.toFixed(2)}x`;
  }

  if (Number.isFinite(median) && median > 1.5) {
    return `Cashout guide: around ${Math.max(1.2, median * 0.8).toFixed(2)}x`;
  }

  return "Cashout guide: high risk, keep target low";
}

function signalStrengthLabel(prediction, direct) {
  if (!prediction) {
    return "Signal: waiting";
  }

  if (prediction.clear_signal && !direct.weak) {
    return `Signal: CLEAR - ${prediction.confidence} confidence`;
  }

  const reason = prediction.clear_reason || "weak signal";
  return `Signal: WEAK - ${reason}`;
}

function bigMultiplierWatch(predictions) {
  const bigTargets = [100, 50, 25, 10];
  const bigPredictions = bigTargets
    .map((target) => findTargetPrediction(predictions, target))
    .filter(Boolean);

  const clearBig = bigPredictions.find(
    (prediction) => prediction.clear_signal && prediction.predicted_high
  );

  if (clearBig) {
    return `Big multiplier watch: ${Number(clearBig.target).toFixed(0)}x+ possible`;
  }

  const strongest = bigPredictions
    .slice()
    .sort((left, right) => Number(right.probability) - Number(left.probability))[0];

  if (!strongest) {
    return "Big multiplier watch: collecting data";
  }

  return `Big multiplier watch: normal (${formatPercent(strongest.probability)} for ${Number(strongest.target).toFixed(0)}x+)`;
}

function formatAccuracyItem(label, item, minimumChecked = 1) {
  if (!item || !item.checked) {
    return `${label} --`;
  }

  if (item.checked < minimumChecked) {
    return `${label} ${item.correct}/${item.checked}`;
  }

  return `${label} ${formatPercent(item.accuracy)}`;
}

function renderAccuracySummary(summary) {
  if (!summary) {
    elements.accuracySummaryText.textContent = "Accuracy: waiting";
    return;
  }

  const range = summary.range || {};
  const rangeText = range.checked
    ? `range ${formatPercent(range.accuracy)} (${range.checked} scored)`
    : "range collecting";
  const skipped = summary.range_skipped ? `${summary.range_skipped} weak skipped` : null;
  const bestTarget = summary.best_target || null;
  const edgeText = bestTarget && bestTarget.skill !== null && bestTarget.skill !== undefined
    ? bestTarget.skill > 0
      ? `best edge >=${Number(bestTarget.target).toFixed(2)}x ${formatSignedPercent(bestTarget.skill)}`
      : "no target beating baseline"
    : "edge waiting";
  const parts = [
    rangeText,
    skipped,
    edgeText,
  ].filter(Boolean);

  elements.accuracySummaryText.textContent = `Accuracy: ${parts.join(" | ")}`;
}

function hasUsefulModelEdge(summary) {
  const bestTarget = summary && summary.best_target ? summary.best_target : null;

  if (!bestTarget || bestTarget.skill === null || bestTarget.skill === undefined) {
    return false;
  }

  return Number(bestTarget.skill) >= 0.02;
}

function shouldShowNoEdge(data, range, direct) {
  const rangeIsClear = Boolean(range && range.clear_signal);
  const directIsClear = Boolean(
    direct
    && !direct.weak
    && direct.prediction
    && direct.prediction.clear_signal
  );

  return !hasUsefulModelEdge(data.accuracy_summary)
    && !rangeIsClear
    && !directIsClear;
}

function renderSourceMode(selection) {
  if (!selection) {
    elements.sourceModeText.textContent = "Data: checking";
    return;
  }

  if (selection.using_trusted_sources) {
    const excluded = selection.excluded_rounds
      ? `, ${selection.excluded_rounds} demo excluded`
      : "";

    elements.sourceModeText.textContent = `Data: real + history (${selection.trusted_rounds} rounds${excluded})`;
    return;
  }

  if (selection.using_source_only) {
    elements.sourceModeText.textContent = `Data: real only (${selection.source_rounds} rounds)`;
    return;
  }

  elements.sourceModeText.textContent = `Data: all rows until ${selection.minimum_source_rounds} real rows (${selection.source_rounds}/${selection.minimum_source_rounds})`;
}

function participantContextParts(participants) {
  const source = String(participants.source || "").toLowerCase();
  const isWorkerTop = source.includes("worker_top");
  const displayCurrency = participants.display_currency || "";
  const betLabel = isWorkerTop
    ? "top streamed bets"
    : "visible bets";
  const betMoneyLabel = isWorkerTop ? "top bet" : "bet";
  const winMoneyLabel = isWorkerTop ? "made" : "made";
  const cashoutLabel = participants.cashed_out_count !== null
    && participants.cashed_out_count !== undefined
    && participants.bet_count !== null
    && participants.bet_count !== undefined
    ? `${participants.cashed_out_count}/${participants.bet_count} cashed out`
    : (
      participants.cashed_out_count !== null && participants.cashed_out_count !== undefined
        ? `${participants.cashed_out_count} cashed out`
        : null
    );

  return [
    participants.bet_count !== null && participants.bet_count !== undefined
      ? `${participants.bet_count} ${betLabel}`
      : betLabel,
    participants.total_bet !== null && participants.total_bet !== undefined
      ? `${betMoneyLabel} ${formatDisplayMoney(displayMoneyValue(participants, "total_bet"), displayCurrency)}`
      : null,
    cashoutLabel,
    participants.avg_cashout !== null && participants.avg_cashout !== undefined
      ? `avg out ${formatCashoutMultiplier(participants.avg_cashout)}`
      : null,
    participants.max_cashout !== null && participants.max_cashout !== undefined
      ? `max out ${formatCashoutMultiplier(participants.max_cashout)}`
      : null,
    participants.total_win !== null && participants.total_win !== undefined
      ? `${winMoneyLabel} ${formatDisplayMoney(displayMoneyValue(participants, "total_win"), displayCurrency)}`
      : null,
  ].filter(Boolean);
}

function renderRoundContext(context) {
  if (!elements.roundContextText) {
    return;
  }

  if (!context || !context.available) {
    elements.roundContextText.textContent = "Context: no bet feed detected yet";
    return;
  }

  const radar = context.radar || (context.source === "flight_radar_dom" ? context : null);
  const participants = context.participants || (context.source === "participants_dom" ? context : null);
  const participantAgeSeconds = participants && participants.age_seconds !== null && participants.age_seconds !== undefined
    ? Number(participants.age_seconds)
    : null;
  const participantsAreFresh = participantAgeSeconds === null
    || participantAgeSeconds <= PARTICIPANT_CONTEXT_FRESH_SECONDS;
  const radarAgeText = radar && radar.age_seconds !== null && radar.age_seconds !== undefined
    ? `updated ${formatAge(radar.age_seconds)} ago`
    : "updated recently";

  if (radar && participants) {
    const participantAgeText = participants.age_seconds !== null && participants.age_seconds !== undefined
      ? `table ${formatAge(participants.age_seconds)} ago`
      : "table recently";
    const players = radar.player_count !== null && radar.player_count !== undefined
      ? `${radar.player_count} live players`
      : "live players";
    const participantParts = participantContextParts(participants);

    if (!participantsAreFresh) {
      elements.roundContextText.textContent = `Context: ${players} | Participants table not live (${participantAgeText}) | open panel for real-time bets/cashouts`;
      return;
    }

    const parts = [
      players,
      ...participantParts,
    ].filter(Boolean);

    elements.roundContextText.textContent = `Context: ${parts.join(" | ")} | ${participantAgeText}`;
    return;
  }

  const ageText = context.age_seconds !== null && context.age_seconds !== undefined
    ? `updated ${formatAge(context.age_seconds)} ago`
    : "updated recently";

  if (radar) {
    const players = radar.player_count !== null && radar.player_count !== undefined
      ? `${radar.player_count} live players`
      : "live players captured";

    elements.roundContextText.textContent = `Context: ${players} | open Participants panel for bet/cashout totals | ${radarAgeText}`;
    return;
  }

  if (participants) {
    const participantAgeText = participants.age_seconds !== null && participants.age_seconds !== undefined
      ? `table ${formatAge(participants.age_seconds)} ago`
      : "table recently";
    const participantParts = participantContextParts(participants);

    if (!participantsAreFresh) {
      elements.roundContextText.textContent = `Context: Participants table not live (${participantAgeText}) | open panel for real-time bets/cashouts`;
      return;
    }

    elements.roundContextText.textContent = `Context: ${participantParts.join(" | ")} | participants | ${ageText}`;
    return;
  }

  const parts = [];

  if (context.player_count !== null && context.player_count !== undefined) {
    parts.push(`${context.player_count} players`);
  }

  if (context.bet_count !== null && context.bet_count !== undefined) {
    parts.push(`${context.bet_count} bets`);
  }

  if (context.total_bet !== null && context.total_bet !== undefined) {
    parts.push(`bet ${formatDisplayMoney(displayMoneyValue(context, "total_bet"), context.display_currency)}`);
  }

  if (context.cashed_out_count !== null && context.cashed_out_count !== undefined) {
    if (context.bet_count !== null && context.bet_count !== undefined) {
      parts.push(`${context.cashed_out_count}/${context.bet_count} cashed out`);
    } else {
      parts.push(`${context.cashed_out_count} cashed out`);
    }
  }

  const roundText = context.round_id ? `round ${context.round_id}` : "latest bet data";

  elements.roundContextText.textContent = `Context: ${parts.join(" | ") || "captured"} | ${roundText} | ${ageText}`;
}

function rangeTone(range, fallbackTone) {
  if (!range) {
    return fallbackTone;
  }

  if (Number(range.minimum) >= 2) {
    return "fast-high";
  }

  if (range.maximum !== null && Number(range.maximum) <= 1.5) {
    return "fast-low";
  }

  return "fast-range";
}

function roundClass(multiplier) {
  if (multiplier >= 10) {
    return "round-extreme";
  }

  if (multiplier >= 5) {
    return "round-high";
  }

  if (multiplier >= 2) {
    return "round-mid";
  }

  return "round-low";
}

function renderPredictionList(predictions) {
  elements.predictionList.innerHTML = "";

  if (!predictions.length) {
    elements.predictionList.textContent = "Waiting for data";
    return;
  }

  for (const prediction of predictions) {
    const row = document.createElement("div");
    const width = Math.max(2, Math.min(100, prediction.probability * 100));
    const direction = prediction.predicted_high ? "MORE" : "LESS";
    const signalText = prediction.clear_signal
      ? `${prediction.signal} signal`
      : prediction.clear_reason || "weak signal";
    row.className = [
      "prediction-item",
      prediction.predicted_high ? "call-high" : "call-low",
      signalClass(prediction.signal),
      targetClass(prediction.target),
    ].join(" ");
    row.innerHTML = `
      <div class="prediction-top">
        <span>${direction} than ${Number(prediction.target).toFixed(2)}x</span>
        <strong>${formatPercent(prediction.probability)}</strong>
      </div>
      <div class="source">
        ${signalText} - ${prediction.confidence} confidence -
        pattern matches ${prediction.evidence.pattern_matches}
      </div>
      <div class="meter"><span style="width: ${width}%; background: ${probabilityColor(prediction.probability)}"></span></div>
    `;
    elements.predictionList.appendChild(row);
  }
}

function renderFastPrediction(data) {
  const predictions = data.next_round.predictions || [];
  const range = data.next_round.range_estimate || null;
  const direct = buildDirectPrediction(predictions);
  const display = buildConsistentDisplay(range, direct);

  elements.fastMain.classList.remove("fast-high", "fast-low", "fast-range", "fast-stale");

  if (data.ingest && data.ingest.is_stale) {
    elements.fastMain.classList.add("fast-stale");
    elements.fastPredictionText.textContent = "DATA STALE";
    elements.fastPredictionMeta.textContent = `Last round ${formatAge(data.ingest.last_round_age_seconds)} ago`;
    elements.signalStrengthText.textContent = "Signal: WAIT - data stale";
    elements.bigWatchText.textContent = "Big multiplier watch: wait for live data";
    elements.fastSignalText.textContent = liveDataLabel(data);
    elements.cashoutGuideText.textContent = liveDataDetail(data, "wait for live data");
    return;
  }

  if (!display) {
    elements.fastPredictionText.textContent = "WAITING";
    elements.fastPredictionMeta.textContent = "Collecting prediction data";
    elements.signalStrengthText.textContent = "Signal: WAIT";
    elements.bigWatchText.textContent = "Big multiplier watch: waiting";
    elements.fastSignalText.textContent = liveDataLabel(data);
    elements.cashoutGuideText.textContent = liveDataDetail(data, "prediction loading");
    return;
  }

  if (shouldShowNoEdge(data, range, direct)) {
    elements.fastMain.classList.add("fast-range");
    elements.fastPredictionText.textContent = "NO CLEAR EDGE";
    elements.fastPredictionMeta.textContent = "Model is not beating baseline yet";
    elements.signalStrengthText.textContent = "Signal: WEAK - no target beating baseline";
    elements.bigWatchText.textContent = bigMultiplierWatch(predictions);
    elements.fastSignalText.textContent = liveDataLabel(data);
    elements.cashoutGuideText.textContent = liveDataDetail(data, "skip or use very low target only");
    return;
  }

  const main = display.prediction;
  elements.fastMain.classList.add(display.tone);
  elements.fastPredictionText.textContent = display.predictionText;
  const lastRoundTime = data.ingest && data.ingest.last_round_timestamp
    ? ` - last ${data.ingest.last_round_timestamp}`
    : "";
  const metaParts = [];

  if (main) {
    metaParts.push(`${formatPercent(main.probability)} chance`);
    metaParts.push(`${main.confidence} confidence`);
  } else if (range) {
    metaParts.push(`${formatPercent(range.probability)} range chance`);
    metaParts.push(`${range.confidence} confidence`);
  }

  elements.signalStrengthText.textContent = display.signalText;
  elements.fastPredictionMeta.textContent = `${metaParts.join(" - ")}${lastRoundTime}`;
  elements.bigWatchText.textContent = bigMultiplierWatch(predictions);
  elements.fastSignalText.textContent = liveDataLabel(data);
  elements.cashoutGuideText.textContent = liveDataDetail(data, display.cashoutText);
}

function renderOverall(probabilities) {
  elements.overallList.innerHTML = "";

  for (const [label, item] of Object.entries(probabilities)) {
    const row = document.createElement("div");
    const width = Math.max(2, Math.min(100, item.probability * 100));
    const target = label.replace(">=", "").replace("x", "");
    row.className = `bar-row ${targetClass(target)}`;
    row.innerHTML = `
      <div class="bar-top">
        <span>${label}</span>
        <strong>${formatPercent(item.probability)}</strong>
      </div>
      <div class="source">${item.hits}/${item.total}</div>
      <div class="meter"><span style="width: ${width}%"></span></div>
    `;
    elements.overallList.appendChild(row);
  }
}

function renderBacktests(backtests) {
  elements.backtestList.innerHTML = "";

  if (!backtests.length) {
    elements.backtestList.textContent = "Waiting for data";
    return;
  }

  for (const item of backtests) {
    const row = document.createElement("div");
    row.className = "table-row";
    row.innerHTML = `
      <span>>=${Number(item.target).toFixed(2)}x</span>
      <strong>${item.accuracy === null ? "n/a" : formatPercent(item.accuracy)}</strong>
      <span>${formatPercent(item.coverage)} coverage - edge ${formatSignedPercent(item.skill)}</span>
    `;
    elements.backtestList.appendChild(row);
  }
}

function renderTracking(tracking) {
  elements.trackingSummary.innerHTML = "";
  const renderTrackingDetails = elements.trackingList.offsetParent !== null;

  if (renderTrackingDetails) {
    elements.trackingList.innerHTML = "";
  }
  elements.previousPredictionStatus.classList.remove("correct", "wrong", "pending");

  if (!tracking) {
    elements.previousPredictionStatus.textContent = "Previous prediction: waiting";
    elements.previousPredictionStatus.classList.add("pending");
    elements.trackingSummary.textContent = "Waiting for prediction tracking";
    return;
  }

  const last = tracking.last_result;

  if (last) {
    const rangeResult = last.range_result || null;
    const correctCount = last.results.filter((item) => item.correct).length;
    const bestResult = last.results
      .slice()
      .sort((left, right) => Math.abs((right.probability || 0) - (right.baseline_probability || 0)) - Math.abs((left.probability || 0) - (left.baseline_probability || 0)))[0];
    const wasSkipped = rangeResult && !rangeResult.scored;
    const weakMatched = wasSkipped
      ? rangeResultMatched(rangeResult, last.actual_multiplier)
      : null;
    const wasCorrect = wasSkipped
      ? null
      : rangeResult
      ? rangeResult.correct
      : bestResult
        ? bestResult.correct
        : correctCount >= Math.ceil(last.results.length / 2);
    elements.previousPredictionStatus.textContent = wasSkipped
      ? "Previous prediction: NOT SCORED"
      : wasCorrect
        ? "Previous prediction: CORRECT"
        : "Previous prediction: WRONG";
    elements.previousPredictionStatus.classList.add(
      wasSkipped ? "pending" : wasCorrect ? "correct" : "wrong"
    );
    const bestText = rangeResult
      ? rangeResult.scored
        ? `predicted range ${rangeResult.display || rangeResult.short || rangeResult.label} was ${rangeResult.correct ? "correct" : "wrong"}`
        : `weak range ${rangeResult.display || rangeResult.short || rangeResult.label} ${weakMatched === null ? "was not scored" : weakMatched ? "matched but was not scored" : "missed but was not scored"} (${rangeResult.clear_reason || "weak range"})`
      : bestResult
        ? `${bestResult.predicted_high ? "more" : "less"} than ${Number(bestResult.target).toFixed(2)}x was ${bestResult.correct ? "correct" : "wrong"}`
        : `${correctCount}/${last.results.length} target calls correct`;
    elements.trackingSummary.innerHTML = `
      Last actual <strong>${formatMultiplier(last.actual_multiplier)}</strong>
      - ${bestText}.
    `;
  } else {
    elements.previousPredictionStatus.textContent = "Previous prediction: pending";
    elements.previousPredictionStatus.classList.add("pending");
    elements.trackingSummary.textContent = "First prediction is pending the next multiplier.";
  }

  const entries = Object.entries(tracking.metrics || {});

  if (!entries.length) {
    if (!renderTrackingDetails) {
      return;
    }

    const row = document.createElement("div");
    row.className = "table-row";
    row.innerHTML = "<span>No checked predictions yet</span><strong>--</strong><span>--</span>";
    elements.trackingList.appendChild(row);
    return;
  }

  for (const [target, metric] of entries) {
    if (!renderTrackingDetails) {
      break;
    }

    const row = document.createElement("div");
    const clearAccuracy = metric.recent_clear_accuracy ?? metric.clear_accuracy;
    const clearText = metric.clear_checked
      ? `clear ${formatPercent(clearAccuracy)}`
      : "clear n/a";
    row.className = "table-row";
    row.innerHTML = `
      <span>>=${Number(target).toFixed(2)}x</span>
      <strong>${formatPercent(metric.recent_accuracy ?? metric.accuracy)}</strong>
      <span>${metric.correct}/${metric.checked} correct - ${clearText}</span>
    `;
    elements.trackingList.appendChild(row);
  }
}

function renderRecentRounds(rounds) {
  elements.recentRounds.innerHTML = "";

  if (!rounds.length) {
    elements.recentRounds.textContent = "Waiting for data";
    return;
  }

  for (const round of rounds.slice(0, 12)) {
    const row = document.createElement("div");
    row.className = `round-row ${roundClass(Number(round.multiplier))}`;
    row.innerHTML = `
      <span>${round.timestamp || "--"}</span>
      <strong>${formatMultiplier(round.multiplier)}</strong>
    `;
    elements.recentRounds.appendChild(row);
  }
}

function drawChart(rounds) {
  const canvas = elements.chart;
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(600, Math.floor(rect.width * ratio));
  canvas.height = Math.max(260, Math.floor(rect.height * ratio));
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

  const width = canvas.width / ratio;
  const height = canvas.height / ratio;
  const padding = { top: 20, right: 16, bottom: 30, left: 44 };

  ctx.clearRect(0, 0, width, height);
  const backgroundGradient = ctx.createLinearGradient(0, 0, width, height);
  backgroundGradient.addColorStop(0, "#151b20");
  backgroundGradient.addColorStop(0.45, "#111418");
  backgroundGradient.addColorStop(1, "#0d0f12");
  ctx.fillStyle = backgroundGradient;
  ctx.fillRect(0, 0, width, height);

  if (!rounds.length) {
    ctx.fillStyle = "#a7b0bb";
    ctx.font = "14px system-ui";
    ctx.fillText("Waiting for rounds", padding.left, height / 2);
    return;
  }

  const values = rounds.map((round) => Number(round.multiplier));
  const cappedValues = values.map((value) => Math.min(value, 25));
  const maxValue = Math.max(5, ...cappedValues);
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  ctx.strokeStyle = "#303640";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#a7b0bb";
  ctx.font = "12px system-ui";

  for (let tick = 0; tick <= 5; tick += 1) {
    const y = padding.top + plotHeight - (plotHeight * tick) / 5;
    const label = ((maxValue * tick) / 5).toFixed(0);
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
    ctx.fillText(`${label}x`, 8, y + 4);
  }

  const points = cappedValues.map((value, index) => {
    const x = padding.left + (plotWidth * index) / Math.max(cappedValues.length - 1, 1);
    const y = padding.top + plotHeight - (plotHeight * value) / maxValue;
    return { x, y };
  });

  const fillGradient = ctx.createLinearGradient(0, padding.top, 0, height - padding.bottom);
  fillGradient.addColorStop(0, "rgba(99, 167, 255, 0.28)");
  fillGradient.addColorStop(0.55, "rgba(78, 215, 209, 0.10)");
  fillGradient.addColorStop(1, "rgba(43, 209, 126, 0.02)");

  ctx.beginPath();
  points.forEach((point, index) => {
    if (index === 0) {
      ctx.moveTo(point.x, point.y);
    } else {
      ctx.lineTo(point.x, point.y);
    }
  });
  ctx.lineTo(points[points.length - 1].x, height - padding.bottom);
  ctx.lineTo(points[0].x, height - padding.bottom);
  ctx.closePath();
  ctx.fillStyle = fillGradient;
  ctx.fill();

  const lineGradient = ctx.createLinearGradient(padding.left, 0, width - padding.right, 0);
  lineGradient.addColorStop(0, "#4ed7d1");
  lineGradient.addColorStop(0.55, "#63a7ff");
  lineGradient.addColorStop(1, "#b8ef5f");
  ctx.strokeStyle = lineGradient;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  points.forEach((point, index) => {
    if (index === 0) {
      ctx.moveTo(point.x, point.y);
    } else {
      ctx.lineTo(point.x, point.y);
    }
  });
  ctx.stroke();

  cappedValues.forEach((value, index) => {
    const { x, y } = points[index];
    ctx.fillStyle = values[index] >= 10 ? "#f3b43f" : values[index] >= 2 ? "#2bd17e" : "#ff5f6d";
    ctx.beginPath();
    ctx.arc(x, y, values[index] >= 10 ? 4.5 : 3, 0, Math.PI * 2);
    ctx.fill();
  });
}

function render(data) {
  const summary = data.summary;
  elements.latestMultiplier.textContent = formatMultiplier(summary.latest_multiplier);
  elements.roundCount.textContent = String(summary.rounds);
  elements.medianValue.textContent = formatMultiplier(summary.median);
  elements.p90Value.textContent = formatMultiplier(summary.p90);
  elements.maxValue.textContent = formatMultiplier(summary.maximum);
  elements.lastUpdated.textContent = data.generated_at || "--";
  elements.riskNote.textContent = data.warning || "";

  const pattern = data.next_round.latest_pattern.length
    ? data.next_round.latest_pattern.join(" -> ")
    : "--";
  elements.latestPattern.textContent = pattern;
  elements.matchCount.textContent = `${data.next_round.pattern_match_count} historical matches`;

  renderFastPrediction(data);
  renderSourceMode(data.data_selection);
  renderRoundContext(data.round_context);
  renderAccuracySummary(data.accuracy_summary);
  if (elements.predictionList.offsetParent !== null) {
    renderPredictionList(data.next_round.predictions);
  }

  if (elements.overallList.offsetParent !== null) {
    renderOverall(data.overall_probabilities || {});
  }

  if (elements.backtestList.offsetParent !== null) {
    renderBacktests(data.backtests || []);
  }

  renderTracking(data.tracking);

  if (elements.recentRounds.offsetParent !== null) {
    renderRecentRounds(data.recent_rounds || []);
  }

  if (elements.chart.offsetParent !== null) {
    drawChart(data.chart_rounds || []);
  }

  const changed = summary.rounds !== state.lastRoundCount;
  state.lastRoundCount = summary.rounds;

  if (data.ingest && data.ingest.is_stale) {
    setStatus("error", `CSV stale - ${formatAge(data.ingest.last_round_age_seconds)}`);
  } else {
    setStatus("live", changed ? "Live - new data" : "Live");
  }
}

function buildRenderSignature(data) {
  const summary = data.summary || {};
  const ingest = data.ingest || {};
  const context = data.round_context || {};
  const radar = context.radar || {};
  const participants = context.participants || {};
  const tracking = data.tracking || {};
  const lastResult = tracking.last_result || {};
  const nextRound = data.next_round || {};
  const range = nextRound.range_estimate || {};
  const predictions = nextRound.predictions || [];

  return JSON.stringify({
    rounds: summary.rounds,
    latest: summary.latest_multiplier,
    age: ingest.last_round_age_seconds,
    radarAt: radar.observed_at || context.observed_at,
    radarPlayers: radar.player_count || context.player_count,
    participantsAt: participants.observed_at,
    participantPlayers: participants.player_count,
    betCount: participants.bet_count,
    totalBet: participants.total_bet,
    displayCurrency: participants.display_currency,
    displayTotalBet: participants.display_total_bet,
    cashouts: participants.cashed_out_count,
    totalWin: participants.total_win,
    displayTotalWin: participants.display_total_win,
    rangeLabel: range.short || range.label,
    rangeSignal: range.clear_signal,
    predictionSignals: predictions.map((item) => [
      item.target,
      item.probability,
      item.predicted_high,
      item.clear_signal,
      item.signal,
    ]),
    lastScore: lastResult.score_id,
    lastActual: lastResult.actual_multiplier,
  });
}

async function refresh() {
  if (state.inFlight) {
    state.pendingRefresh = true;
    return;
  }

  state.inFlight = true;

  const params = new URLSearchParams({
    lookback: elements.lookbackSelect.value,
    min_matches: elements.minMatchesSelect.value,
    t: Date.now().toString(),
  });

  try {
    const response = await fetch(`${LIVE_ENDPOINT}?${params.toString()}`, {
      cache: "no-store",
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    const signature = buildRenderSignature(data);

    if (signature !== state.lastRenderSignature) {
      state.lastRenderSignature = signature;
      render(data);
    }
  } catch (error) {
    setStatus("error", "Disconnected");
    console.error(error);
  } finally {
    state.inFlight = false;

    if (state.pendingRefresh) {
      state.pendingRefresh = false;
      refresh();
    }
  }
}

function start() {
  elements.refreshButton.addEventListener("click", refresh);
  elements.lookbackSelect.addEventListener("change", refresh);
  elements.minMatchesSelect.addEventListener("change", refresh);
  window.addEventListener("resize", () => refresh());

  refresh();
  state.timer = window.setInterval(refresh, REFRESH_MS);
}

start();
