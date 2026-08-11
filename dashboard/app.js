const state = {
  timer: null,
  lastRoundCount: 0,
  inFlight: false,
  pendingRefresh: false,
  lastRenderSignature: "",
  lastBigRoundKey: "",
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
  selfLearningText: document.getElementById("selfLearningText"),
  sourceModeText: document.getElementById("sourceModeText"),
  roundContextText: document.getElementById("roundContextText"),
  bigRoundPanel: document.getElementById("bigRoundPanel"),
  bigRoundStatus: document.getElementById("bigRoundStatus"),
  bigRoundList: document.getElementById("bigRoundList"),
  mlPredictionPanel: document.getElementById("mlPredictionPanel"),
  mlModelStatus: document.getElementById("mlModelStatus"),
  mlProbabilityList: document.getElementById("mlProbabilityList"),
  mlModelMeta: document.getElementById("mlModelMeta"),
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

function formatPercentagePoints(value) {
  if (value === null || value === undefined) {
    return "--";
  }

  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "--";
  }

  const sign = number > 0 ? "+" : "";
  return `${sign}${(number * 100).toFixed(1)} pp`;
}

function formatContextNumber(value) {
  if (value === null || value === undefined) {
    return "--";
  }

  return Number(value).toFixed(2);
}

function formatCount(value) {
  if (value === null || value === undefined) {
    return "--";
  }

  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "--";
  }

  return Math.round(number).toLocaleString("en-IN");
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

function formatSignedDisplayMoney(value, currency) {
  if (value === null || value === undefined) {
    return "--";
  }

  const amount = Number(value);

  if (!Number.isFinite(amount)) {
    return "--";
  }

  const sign = amount > 0 ? "+" : amount < 0 ? "-" : "";
  return `${sign}${formatDisplayMoney(Math.abs(amount), currency)}`;
}

function displayMoneyValue(context, key) {
  const displayKey = `display_${key}`;

  if (context[displayKey] !== null && context[displayKey] !== undefined) {
    return context[displayKey];
  }

  return context[key];
}

function contextNetValue(context) {
  if (!context) {
    return null;
  }

  const direct = displayMoneyValue(context, "net_result");

  if (direct !== null && direct !== undefined) {
    return direct;
  }

  const totalWin = displayMoneyValue(context, "total_win");
  const totalBet = displayMoneyValue(context, "total_bet");

  if (totalWin === null || totalWin === undefined || totalBet === null || totalBet === undefined) {
    return null;
  }

  const net = Number(totalWin) - Number(totalBet);
  return Number.isFinite(net) ? net : null;
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
  if (data.ingest && data.ingest.is_stale) {
    return "PAUSED";
  }

  return "LIVE";
}

function cleanCashoutGuide(text) {
  return String(text || "waiting").replace(/^Cash out:\s*/i, "");
}

function liveDataDetail(data, cashoutText) {
  const ageText = data.ingest
    ? `Last round ${formatAge(data.ingest.last_round_age_seconds)} ago`
    : "Last round unknown";
  const selection = data.data_selection || {};
  const roundCount = selection.using_trusted_sources
    ? `${selection.trusted_rounds} rounds saved`
    : selection.using_source_only
    ? `${selection.source_rounds} rounds saved`
    : `${data.summary ? data.summary.rounds : "--"} rounds`;

  return `${ageText} | ${roundCount} | ${cleanCashoutGuide(cashoutText)}`;
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
      text: "Above 100.00x",
      note: "Very high round possible",
    };
  }

  if (p50 && p50.predicted_high && p50.signal === "FAVOR") {
    return {
      prediction: p50,
      tone: "fast-high",
      text: "Above 50.00x",
      note: "Big round possible",
    };
  }

  if (p25 && p25.predicted_high && p25.signal === "FAVOR") {
    return {
      prediction: p25,
      tone: "fast-high",
      text: "Above 25.00x",
      note: "High round possible",
    };
  }

  if (p10 && p10.predicted_high && p10.signal === "FAVOR") {
    return {
      prediction: p10,
      tone: "fast-high",
      text: "Above 10.00x",
      note: "10x round possible",
    };
  }

  if (p5 && p5.predicted_high && p5.signal === "FAVOR") {
    return {
      prediction: p5,
      tone: "fast-high",
      text: "Above 5.00x",
      note: "Higher round possible",
    };
  }

  if (p3 && p3.predicted_high && p3.signal === "FAVOR") {
    return {
      prediction: p3,
      tone: "fast-high",
      text: "Above 3.00x",
      note: "3x round possible",
    };
  }

  if (p2 && p2.predicted_high) {
    return {
      prediction: p2,
      tone: "fast-high",
      text: "Above 2.00x",
      note: "2x round possible",
    };
  }

  if (low15 && !low15.predicted_high) {
    return {
      prediction: low15,
      tone: "fast-low",
      text: "Below 1.50x",
      note: "Low round risk",
    };
  }

  if (low15 && low15.predicted_high && !findTargetPrediction(clearPredictions, 2)) {
    return {
      prediction: low15,
      tone: "fast-range",
      text: "Between 1.50x and 2.00x",
      note: "Small round expected",
    };
  }

  if (fallback) {
    return {
      prediction: fallback,
      tone: fallback.predicted_high ? "fast-high" : "fast-low",
      text: fallback.predicted_high
        ? `Above ${Number(fallback.target).toFixed(2)}x`
        : `Below ${Number(fallback.target).toFixed(2)}x`,
      note: "Best current call",
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
        ? `Above ${Number(strongestWeak.target).toFixed(2)}x`
        : `Below ${Number(strongestWeak.target).toFixed(2)}x`,
      note: strongestWeak && strongestWeak.clear_reason
        ? strongestWeak.clear_reason
        : "Not strong enough yet",
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

  if (range.maximum === null || range.maximum === undefined) {
    return `Expected above ${Number(range.minimum).toFixed(2)}x`;
  }

  return `Expected ${Number(range.minimum).toFixed(2)}x to ${Number(range.maximum).toFixed(2)}x`;
}

function formatMainPrediction(range, direct) {
  if (
    direct
    && !direct.weak
    && direct.prediction
    && direct.prediction.predicted_high
    && Number(direct.prediction.target) >= 2
  ) {
    return `Expected above ${Number(direct.prediction.target).toFixed(2)}x`;
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
    return `above ${Number(range.minimum).toFixed(2)}x`;
  }

  return `${Number(range.minimum).toFixed(2)}x to ${Number(range.maximum).toFixed(2)}x`;
}

function coverageRangeText(range) {
  if (!range) {
    return "";
  }

  return `Safer area: ${rangeCallText(range)} (${formatPercent(range.probability)} chance)`;
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
        ? `Above ${Number(direct.prediction.target).toFixed(2)}x`
        : `Below ${Number(direct.prediction.target).toFixed(2)}x`,
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
        ? `Confidence: ${range.confidence.toUpperCase()}`
        : `Confidence: LOW - play safe`,
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
        ? `Above ${Number(direct.prediction.target).toFixed(2)}x`
        : `Below ${Number(direct.prediction.target).toFixed(2)}x`,
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

  if (range.target_confidence !== null && range.target_confidence !== undefined) {
    return `Chance ${formatPercent(range.probability)} for this wider area`;
  }

  return `Chance ${formatPercent(range.probability)} | typical area ${formatMultiplier(range.low)} to ${formatMultiplier(range.high)}`;
}

function cashoutGuide(range, direct) {
  if (
    direct
    && !direct.weak
    && direct.prediction
    && direct.prediction.predicted_high
    && Number(direct.prediction.target) >= 2
  ) {
    return `Cash out: before ${Number(direct.prediction.target).toFixed(2)}x`;
  }

  if (!range) {
    return "Cash out: waiting";
  }

  if (range.clear_signal === false) {
    if (range.cashout_target !== null && range.cashout_target !== undefined) {
      return `Cash out: around ${Number(range.cashout_target).toFixed(2)}x`;
    }

    return "Cash out: keep target low";
  }

  if (range.cashout_target !== null && range.cashout_target !== undefined) {
    return `Cash out: before ${Number(range.cashout_target).toFixed(2)}x`;
  }

  const minimum = Number(range.minimum);
  const maximum = range.maximum === null || range.maximum === undefined
    ? null
    : Number(range.maximum);
  const median = Number(range.median);

  if (maximum !== null && maximum <= 1.5) {
    return `Cash out: very early, before ${maximum.toFixed(2)}x`;
  }

  if (maximum !== null && maximum <= 2) {
    return `Cash out: before ${Math.max(1.2, minimum).toFixed(2)}x`;
  }

  if (minimum >= 5) {
    return `Cash out: protect before ${minimum.toFixed(2)}x`;
  }

  if (minimum >= 2) {
    return `Cash out: before ${minimum.toFixed(2)}x`;
  }

  if (Number.isFinite(median) && median > 1.5) {
    return `Cash out: around ${Math.max(1.2, median * 0.8).toFixed(2)}x`;
  }

  return "Cash out: high risk, keep target low";
}

function signalStrengthLabel(prediction, direct) {
  if (!prediction) {
    return "Confidence: waiting";
  }

  if (prediction.clear_signal && !direct.weak) {
    return `Confidence: ${prediction.confidence.toUpperCase()}`;
  }

  return "Confidence: LOW - play safe";
}

function bigMultiplierWatch(predictions) {
  const bigTargets = [100, 50, 20, 10];
  const bigPredictions = bigTargets
    .map((target) => findTargetPrediction(predictions, target))
    .filter(Boolean);

  const clearBig = bigPredictions.find(
    (prediction) => prediction.clear_signal && prediction.predicted_high
  );

  if (clearBig) {
    return `High round chance: ${Number(clearBig.target).toFixed(0)}x+ possible`;
  }

  const strongest = bigPredictions
    .slice()
    .sort((left, right) => Number(right.probability) - Number(left.probability))[0];

  if (!strongest) {
    return "High round chance: waiting";
  }

  return `High round chance: normal (${formatPercent(strongest.probability)} for ${Number(strongest.target).toFixed(0)}x+)`;
}

function mlPredictionEntries(mlPrediction) {
  return Object.entries((mlPrediction && mlPrediction.predictions) || {})
    .map(([target, item]) => ({
      target: Number(target),
      ...(item || {}),
    }))
    .filter((item) => Number.isFinite(item.target))
    .sort((left, right) => left.target - right.target);
}

function mlTarget(mlPrediction, target) {
  return mlPredictionEntries(mlPrediction).find(
    (item) => Number(item.target).toFixed(2) === Number(target).toFixed(2)
  );
}

function mlHasProvenEdge(mlPrediction) {
  return mlPredictionEntries(mlPrediction).some((item) => {
    const status = String(item.holdout_status || item.validation_status || "");
    const edge = Math.abs(Number(item.edge || 0));
    const brierSkill = Number(item.holdout_brier_skill || 0);
    return status.includes("CONSISTENT") || (edge >= 0.02 && brierSkill > 0);
  });
}

function mlStatusLabel(mlPrediction) {
  if (!mlPrediction) {
    return "ML: waiting";
  }

  if (!mlPrediction.available) {
    return "ML: not ready";
  }

  if (mlHasProvenEdge(mlPrediction)) {
    return "ML: edge detected";
  }

  return "ML: no proven edge";
}

function mlBigWatchText(mlPrediction) {
  const p10 = mlTarget(mlPrediction, 10);

  if (!p10) {
    return "10x chance: waiting";
  }

  return `10x chance: ${formatPercent(p10.probability)}`;
}

function buildMlFastDisplay(mlPrediction) {
  if (!mlPrediction || !mlPrediction.available) {
    return null;
  }

  const p15 = mlTarget(mlPrediction, 1.5);
  const p2 = mlTarget(mlPrediction, 2);
  const p3 = mlTarget(mlPrediction, 3);
  const p5 = mlTarget(mlPrediction, 5);
  const p10 = mlTarget(mlPrediction, 10);
  const entries = mlPredictionEntries(mlPrediction);

  if (!entries.length) {
    return null;
  }

  const hasEdge = mlHasProvenEdge(mlPrediction);
  const bestEdge = entries
    .slice()
    .sort((left, right) => Math.abs(Number(right.edge || 0)) - Math.abs(Number(left.edge || 0)))[0];
  const main = p2 || p15 || entries[0];
  const probabilityLine = [
    p15 ? `1.5x ${formatPercent(p15.probability)}` : null,
    p2 ? `2x ${formatPercent(p2.probability)}` : null,
    p3 ? `3x ${formatPercent(p3.probability)}` : null,
    p5 ? `5x ${formatPercent(p5.probability)}` : null,
    p10 ? `10x ${formatPercent(p10.probability)}` : null,
  ].filter(Boolean).join(" | ");

  if (!hasEdge) {
    return {
      tone: "fast-range",
      text: `${Number(main.target).toFixed(0)}x+ chance ${formatPercent(main.probability)}`,
      meta: `${probabilityLine} | same as history`,
      signal: "Model status: no proven edge",
      cashout: "ML has no reliable cashout target",
      bigWatch: mlBigWatchText(mlPrediction),
    };
  }

  return {
    tone: Number(bestEdge.target) >= 2 ? "fast-high" : "fast-range",
    text: `${Number(bestEdge.target).toFixed(0)}x+ chance ${formatPercent(bestEdge.probability)}`,
    meta: `Edge ${formatPercentagePoints(bestEdge.edge)} vs history | ${probabilityLine}`,
    signal: `Model status: ${bestEdge.holdout_status || bestEdge.validation_status || "active"}`,
    cashout: `ML target: ${Number(bestEdge.target).toFixed(2)}x area`,
    bigWatch: mlBigWatchText(mlPrediction),
  };
}

function renderMlPrediction(mlPrediction) {
  if (
    !elements.mlPredictionPanel
    || !elements.mlModelStatus
    || !elements.mlProbabilityList
    || !elements.mlModelMeta
  ) {
    return;
  }

  elements.mlPredictionPanel.classList.toggle(
    "has-edge",
    mlHasProvenEdge(mlPrediction),
  );
  elements.mlModelStatus.textContent = mlStatusLabel(mlPrediction);
  elements.mlProbabilityList.innerHTML = "";

  if (!mlPrediction || !mlPrediction.available) {
    elements.mlProbabilityList.textContent = mlPrediction && mlPrediction.error
      ? mlPrediction.error
      : "Waiting for trained model";
    elements.mlModelMeta.textContent = "Run ml_train.py once if model files are missing.";
    return;
  }

  const visibleEntries = mlPredictionEntries(mlPrediction).filter(
    (item) => [1.5, 2, 3, 5, 10].includes(Number(item.target))
  );

  for (const item of visibleEntries) {
    const row = document.createElement("div");
    row.className = "ml-probability-row";
    row.innerHTML = `
      <span>${Number(item.target).toFixed(item.target === 10 ? 0 : 1)}x+</span>
      <strong>${formatPercent(item.probability)}</strong>
      <small>${formatPercentagePoints(item.edge)} vs history</small>
    `;
    elements.mlProbabilityList.appendChild(row);
  }

  const modelNames = [...new Set(visibleEntries.map((item) => item.model).filter(Boolean))];
  const roundsText = mlPrediction.data_used_rounds
    ? `${formatCount(mlPrediction.data_used_rounds)} rounds`
    : "round count unknown";
  const currentText = mlPrediction.is_current ? "current" : "refreshing";
  elements.mlModelMeta.textContent = `${roundsText} | ${modelNames.join(", ") || "model"} | ${currentText}`;
}

function formatRoundsAgo(value) {
  if (value === null || value === undefined) {
    return "unknown";
  }

  const roundsAgo = Number(value);

  if (!Number.isFinite(roundsAgo)) {
    return "unknown";
  }

  if (roundsAgo <= 0) {
    return "just now";
  }

  if (roundsAgo === 1) {
    return "1 round ago";
  }

  return `${roundsAgo} rounds ago`;
}

function formatAverageGap(rate) {
  if (rate === null || rate === undefined) {
    return "average gap unknown";
  }

  const probability = Number(rate);

  if (!Number.isFinite(probability) || probability <= 0) {
    return "average gap unknown";
  }

  const gap = Math.max(1, Math.round(1 / probability));
  return `average gap ${gap} rounds`;
}

function bigRoundGapText(item) {
  const last = item.last || null;
  const lastText = last ? formatRoundsAgo(last.rounds_ago) : "not seen";
  return `last ${lastText} | ${formatAverageGap(item.rate)}`;
}

function bigRoundKey(event) {
  if (!event) {
    return "";
  }

  return [
    event.round_number || "",
    event.timestamp || "",
    event.multiplier || "",
  ].join("|");
}

function renderBigRounds(bigRounds) {
  if (!elements.bigRoundPanel || !elements.bigRoundStatus || !elements.bigRoundList) {
    return;
  }

  elements.bigRoundPanel.classList.remove("fresh");
  elements.bigRoundList.innerHTML = "";

  if (!bigRounds || !Array.isArray(bigRounds.thresholds)) {
    elements.bigRoundStatus.textContent = "Waiting";
    elements.bigRoundList.textContent = "Collecting big round history";
    return;
  }

  const latest = bigRounds.latest || null;

  if (!latest) {
    elements.bigRoundStatus.textContent = "No 10x+ yet";
  } else {
    elements.bigRoundStatus.textContent = `${formatMultiplier(latest.multiplier)} ${formatRoundsAgo(latest.rounds_ago)}`;
  }

  const latestKey = bigRoundKey(latest);

  if (!state.lastBigRoundKey && latestKey) {
    state.lastBigRoundKey = latestKey;
  } else if (latestKey && latestKey !== state.lastBigRoundKey) {
    state.lastBigRoundKey = latestKey;

    if (latest && Number(latest.rounds_ago) === 0) {
      elements.bigRoundPanel.classList.add("fresh");
      window.setTimeout(() => {
        elements.bigRoundPanel.classList.remove("fresh");
      }, 2400);
    }
  }

  for (const item of bigRounds.thresholds) {
    const target = Number(item.target);
    const row = document.createElement("div");
    row.className = "big-round-row";
    row.innerHTML = `
      <span>${target.toFixed(0)}x+</span>
      <strong>${formatCount(item.count)}</strong>
      <small>${bigRoundGapText(item)}</small>
    `;
    elements.bigRoundList.appendChild(row);
  }
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
    elements.accuracySummaryText.textContent = "Score: waiting";
    elements.selfLearningText.textContent = "Learning: waiting";
    return;
  }

  const range = summary.range || {};
  const rangeText = range.checked
    ? `Last ${range.checked} checks: ${range.correct}/${range.checked} right`
    : "Score: collecting checks";
  const bestTarget = summary.best_target || null;
  const targetText = bestTarget && bestTarget.checked
    ? `Best simple call: ${formatPercent(bestTarget.accuracy)} for ${Number(bestTarget.target).toFixed(2)}x+`
    : "Best simple call: waiting";
  const parts = [
    rangeText,
    targetText,
  ].filter(Boolean);

  elements.accuracySummaryText.textContent = parts.join(" | ");
  renderSelfLearningStatus(summary.self_learning);
}

function renderSelfLearningStatus(status) {
  if (!status || !status.enabled) {
    elements.selfLearningText.textContent = "Learning: off";
    return;
  }

  if (status.status === "active" && status.active_model) {
    elements.selfLearningText.textContent = `Learning: using best method (${formatPercent(status.active_model.accuracy)} over ${status.active_model.checked} checks)`;
    return;
  }

  const best = status.best_model;
  const bestText = best
    ? `best so far ${formatPercent(best.accuracy)}`
    : "checking methods";
  const remaining = Number(status.rounds_until_auto_select || 0);
  const confirmText = remaining > 0
    ? `${remaining} more rounds to confirm`
    : "waiting for better score";
  elements.selfLearningText.textContent = `Learning: ${bestText} | ${confirmText}`;
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
  const rangeMaximum = range && range.maximum !== null && range.maximum !== undefined
    ? Number(range.maximum)
    : null;
  const rangeIsNarrowEstimate = Boolean(
    range
    && rangeMaximum !== null
    && Number.isFinite(rangeMaximum)
    && rangeMaximum <= 3
  );
  const directIsClear = Boolean(
    direct
    && !direct.weak
    && direct.prediction
    && direct.prediction.clear_signal
  );

  return !hasUsefulModelEdge(data.accuracy_summary)
    && !rangeIsClear
    && !rangeIsNarrowEstimate
    && !directIsClear;
}

function renderSourceMode(selection) {
  if (!selection) {
    elements.sourceModeText.textContent = "Data: checking";
    return;
  }

  if (selection.using_trusted_sources) {
    elements.sourceModeText.textContent = `Data: ${formatCount(selection.trusted_rounds)} rounds saved`;
    return;
  }

  if (selection.using_source_only) {
    elements.sourceModeText.textContent = `Data: ${formatCount(selection.source_rounds)} rounds saved`;
    return;
  }

  elements.sourceModeText.textContent = `Data: ${formatCount(selection.source_rounds)} game rounds saved`;
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

function participantSourceLabel(participants) {
  const source = String(participants && participants.source || "").toLowerCase();

  if (source.includes("worker_top")) {
    return "live bet rows";
  }

  if (source.includes("participants_dom")) {
    return "visible bet rows";
  }

  if (source.includes("worker_active")) {
    return "live player count";
  }

  if (source.includes("userbets")) {
    return "bet feed";
  }

  if (source.includes("participants")) {
    return "players feed";
  }

  return "live feed";
}

function contextMetric(label, value, detail, tone) {
  const item = document.createElement("span");
  item.className = `context-card${tone ? ` ${tone}` : ""}`;

  const labelNode = document.createElement("em");
  labelNode.textContent = label;

  const valueNode = document.createElement("strong");
  valueNode.textContent = value;

  const detailNode = document.createElement("small");
  detailNode.textContent = detail;

  item.append(labelNode, valueNode, detailNode);
  return item;
}

function contextNetMetric(context) {
  const net = contextNetValue(context);
  const displayCurrency = context && context.display_currency;
  const tone = net === null
    ? "money"
    : net > 0
      ? "profit"
      : net < 0
        ? "loss"
        : "money";
  const detail = net === null
    ? "waiting for totals"
    : net > 0
      ? "players ahead"
      : net < 0
        ? "players down"
        : "even";

  return contextMetric(
    "Player result",
    net === null ? "--" : formatSignedDisplayMoney(net, displayCurrency),
    detail,
    tone,
  );
}

function setRoundContextCards(cards, footerText) {
  elements.roundContextText.replaceChildren();
  elements.roundContextText.classList.add("context-cards");

  for (const card of cards) {
    elements.roundContextText.appendChild(card);
  }

  if (footerText) {
    const footer = document.createElement("span");
    footer.className = "context-footer";
    footer.textContent = footerText;
    elements.roundContextText.appendChild(footer);
  }
}

function renderRoundContext(context) {
  if (!elements.roundContextText) {
    return;
  }

  elements.roundContextText.classList.remove("context-cards");

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
  const activePlayerCount = Math.max(
    radar && radar.player_count !== null && radar.player_count !== undefined ? Number(radar.player_count) : 0,
    participants && participants.player_count !== null && participants.player_count !== undefined ? Number(participants.player_count) : 0,
    context.player_count !== null && context.player_count !== undefined ? Number(context.player_count) : 0,
  );
  const contextAgeSeconds = participants && participants.age_seconds !== null && participants.age_seconds !== undefined
    ? participants.age_seconds
    : context.age_seconds;
  const contextAgeText = contextAgeSeconds !== null && contextAgeSeconds !== undefined
    ? `${formatAge(contextAgeSeconds)} ago`
    : "recent";

  if (radar && participants) {
    const participantAgeText = participants.age_seconds !== null && participants.age_seconds !== undefined
      ? `${formatAge(participants.age_seconds)} ago`
      : "recent";
    const capturedLabel = participantSourceLabel(participants);
    const capturedCount = participants.bet_count !== null && participants.bet_count !== undefined
      ? formatCount(participants.bet_count)
      : "--";
    const cashoutPercent = participants.cashed_out_count !== null
      && participants.cashed_out_count !== undefined
      && participants.bet_count
      ? ` (${formatPercent(Number(participants.cashed_out_count) / Number(participants.bet_count))})`
      : "";
    const cards = [
      contextMetric("Players", activePlayerCount ? formatCount(activePlayerCount) : "--", "live now", "primary"),
      contextMetric("Bets seen", capturedCount, capturedLabel, "info"),
      contextMetric(
        "Bet amount",
        participants.total_bet !== null && participants.total_bet !== undefined
          ? formatDisplayMoney(displayMoneyValue(participants, "total_bet"), participants.display_currency)
          : "--",
        "seen on screen/feed",
        "money",
      ),
      contextMetric(
        "Cashed out",
        participants.cashed_out_count !== null && participants.cashed_out_count !== undefined
          ? `${formatCount(participants.cashed_out_count)}/${capturedCount}`
          : "--",
        cashoutPercent ? `seen bets${cashoutPercent}` : "seen bets",
        "cashout",
      ),
      contextNetMetric(participants),
      contextMetric(
        "Paid out",
        participants.total_win !== null && participants.total_win !== undefined
          ? formatDisplayMoney(displayMoneyValue(participants, "total_win"), participants.display_currency)
          : "--",
        "seen cashouts",
        "money",
      ),
      contextMetric("Updated", participantAgeText, participantsAreFresh ? "live" : "not live", participantsAreFresh ? "fresh" : "stale"),
    ];

    if (!participantsAreFresh) {
      setRoundContextCards(
        cards,
        "Open Participants if bet/cashout numbers stop moving.",
      );
      return;
    }

    setRoundContextCards(
      cards,
      "Bet amounts are from visible/live rows, not necessarily every player.",
    );
    return;
  }

  const ageText = context.age_seconds !== null && context.age_seconds !== undefined
    ? `updated ${formatAge(context.age_seconds)} ago`
    : "updated recently";

  if (radar) {
    setRoundContextCards(
      [
        contextMetric("Players", radar.player_count !== null && radar.player_count !== undefined ? formatCount(radar.player_count) : "--", "live now", "primary"),
        contextMetric("Bets seen", "--", "open Participants", "stale"),
        contextMetric("Updated", radarAgeText.replace("updated ", ""), "player count", "fresh"),
      ],
      "Bet totals appear after the Participants panel/feed is detected.",
    );
    return;
  }

  if (participants) {
    const participantAgeText = participants.age_seconds !== null && participants.age_seconds !== undefined
      ? `${formatAge(participants.age_seconds)} ago`
      : "recent";
    const capturedCount = participants.bet_count !== null && participants.bet_count !== undefined
      ? formatCount(participants.bet_count)
      : "--";
    const cards = [
      contextMetric("Players", activePlayerCount ? formatCount(activePlayerCount) : "--", "live now", "primary"),
      contextMetric("Bets seen", capturedCount, participantSourceLabel(participants), "info"),
      contextMetric(
        "Bet amount",
        participants.total_bet !== null && participants.total_bet !== undefined
          ? formatDisplayMoney(displayMoneyValue(participants, "total_bet"), participants.display_currency)
          : "--",
        "seen bets",
        "money",
      ),
      contextMetric(
        "Cashed out",
        participants.cashed_out_count !== null && participants.cashed_out_count !== undefined
          ? `${formatCount(participants.cashed_out_count)}/${capturedCount}`
          : "--",
        "seen bets",
        "cashout",
      ),
      contextNetMetric(participants),
      contextMetric("Updated", participantAgeText, participantsAreFresh ? "live" : "not live", participantsAreFresh ? "fresh" : "stale"),
    ];

    if (!participantsAreFresh) {
      setRoundContextCards(
        cards,
        "Open Participants if bet/cashout numbers stop moving.",
      );
      return;
    }

    setRoundContextCards(
      cards,
      "Bet amounts are from visible/live rows.",
    );
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

  setRoundContextCards(
    [
      contextMetric("Players", context.player_count !== null && context.player_count !== undefined ? formatCount(context.player_count) : "--", "live now", "primary"),
      contextMetric("Bets", context.bet_count !== null && context.bet_count !== undefined ? formatCount(context.bet_count) : "--", roundText, "info"),
      contextMetric("Bet amount", context.total_bet !== null && context.total_bet !== undefined ? formatDisplayMoney(displayMoneyValue(context, "total_bet"), context.display_currency) : "--", "seen bets", "money"),
      contextNetMetric(context),
      contextMetric("Updated", contextAgeText, ageText.replace("updated ", ""), "fresh"),
    ],
    parts.join(" | ") || "Latest live details.",
  );
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
  const coverageRange = range && range.coverage_range ? range.coverage_range : null;
  const direct = buildDirectPrediction(predictions);
  const display = buildConsistentDisplay(range, direct);

  elements.fastMain.classList.remove("fast-high", "fast-low", "fast-range", "fast-stale");

  if (data.ingest && data.ingest.is_stale) {
    elements.fastMain.classList.add("fast-stale");
    elements.fastPredictionText.textContent = "Waiting for live data";
    elements.fastPredictionMeta.textContent = `Last round ${formatAge(data.ingest.last_round_age_seconds)} ago`;
    elements.signalStrengthText.textContent = "Confidence: paused";
    elements.bigWatchText.textContent = "High round chance: waiting";
    elements.fastSignalText.textContent = liveDataLabel(data);
    elements.cashoutGuideText.textContent = liveDataDetail(data, "wait for live data");
    return;
  }

  const mlDisplay = buildMlFastDisplay(data.ml_prediction);

  if (mlDisplay) {
    elements.fastMain.classList.add(mlDisplay.tone);
    elements.fastPredictionText.textContent = mlDisplay.text;
    elements.fastPredictionMeta.textContent = mlDisplay.meta;
    elements.signalStrengthText.textContent = mlDisplay.signal;
    elements.bigWatchText.textContent = mlDisplay.bigWatch;
    elements.fastSignalText.textContent = liveDataLabel(data);
    elements.cashoutGuideText.textContent = liveDataDetail(data, mlDisplay.cashout);
    return;
  }

  if (!display) {
    elements.fastPredictionText.textContent = "Waiting";
    elements.fastPredictionMeta.textContent = "Collecting prediction data";
    elements.signalStrengthText.textContent = "Confidence: waiting";
    elements.bigWatchText.textContent = "High round chance: waiting";
    elements.fastSignalText.textContent = liveDataLabel(data);
    elements.cashoutGuideText.textContent = liveDataDetail(data, "prediction loading");
    return;
  }

  if (shouldShowNoEdge(data, range, direct)) {
    elements.fastMain.classList.add("fast-range");
    elements.fastPredictionText.textContent = "Wait for a clearer call";
    elements.fastPredictionMeta.textContent = coverageRange
      ? coverageRangeText(coverageRange)
      : "No strong pattern right now";
    elements.signalStrengthText.textContent = "Confidence: LOW - play safe";
    elements.bigWatchText.textContent = bigMultiplierWatch(predictions);
    elements.fastSignalText.textContent = liveDataLabel(data);
    elements.cashoutGuideText.textContent = liveDataDetail(data, "no reliable cashout target");
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
    metaParts.push(`Chance ${formatPercent(main.probability)}`);
    metaParts.push(`Confidence ${main.confidence}`);
  } else if (range) {
    metaParts.push(
      range.target_confidence !== null && range.target_confidence !== undefined
        ? `Chance ${formatPercent(range.probability)}`
        : `Chance ${formatPercent(range.probability)}`
    );
    metaParts.push(`Confidence ${range.confidence}`);

    if (coverageRange && range.range_type !== "confidence_80") {
      metaParts.push(`Safer area ${rangeCallText(coverageRange)}`);
    }
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
      <span>${formatPercent(item.coverage)} seen in history</span>
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
    elements.previousPredictionStatus.textContent = "Waiting";
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
      ? "Not checked"
      : wasCorrect
        ? "Right"
        : "Wrong";
    elements.previousPredictionStatus.classList.add(
      wasSkipped ? "pending" : wasCorrect ? "correct" : "wrong"
    );
    const bestText = rangeResult
      ? rangeResult.scored
        ? `Expected ${rangeResult.display || rangeResult.short || rangeResult.label}; it was ${rangeResult.correct ? "inside" : "outside"} that range`
        : `Last call was low confidence, so it was not counted${weakMatched === null ? "" : weakMatched ? " (it still matched)" : " (it missed)"}`
      : bestResult
        ? `${bestResult.predicted_high ? "Above" : "Below"} ${Number(bestResult.target).toFixed(2)}x was ${bestResult.correct ? "right" : "wrong"}`
        : `${correctCount}/${last.results.length} quick checks were right`;
    elements.trackingSummary.innerHTML = `
      Last crash <strong>${formatMultiplier(last.actual_multiplier)}</strong>.
      ${bestText}.
    `;
  } else {
    elements.previousPredictionStatus.textContent = "Pending";
    elements.previousPredictionStatus.classList.add("pending");
    elements.trackingSummary.textContent = "Waiting for the next crash to check it.";
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
  renderBigRounds(data.big_rounds);
  renderMlPrediction(data.ml_prediction);
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
  const coverageRange = range.coverage_range || {};
  const predictions = nextRound.predictions || [];
  const bigRounds = data.big_rounds || {};
  const latestBig = bigRounds.latest || {};
  const accuracy = data.accuracy_summary || {};
  const activeRangeModel = accuracy.active_range_model || {};
  const selfLearning = accuracy.self_learning || {};
  const bestLearningModel = selfLearning.best_model || {};
  const mlPrediction = data.ml_prediction || {};
  const mlEntries = mlPredictionEntries(mlPrediction);

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
    netResult: participants.net_result,
    displayNetResult: participants.display_net_result,
    latestBigRound: latestBig.round_number,
    latestBigMultiplier: latestBig.multiplier,
    latestBigRoundsAgo: latestBig.rounds_ago,
    bigThresholds: (bigRounds.thresholds || []).map((item) => [
      item.target,
      item.count,
      item.last ? item.last.round_number : null,
    ]),
    rangeLabel: range.short || range.label,
    rangeSignal: range.clear_signal,
    rangeTarget: range.target_confidence,
    rangeCashout: range.cashout_target,
    rangeProbability: range.probability,
    rangeReason: range.clear_reason,
    coverageLabel: coverageRange.short || coverageRange.label,
    coverageProbability: coverageRange.probability,
    coverageReason: coverageRange.clear_reason,
    activeRangeModel: activeRangeModel.candidate_model,
    activeRangeAccuracy: activeRangeModel.accuracy,
    activeRangeChecked: activeRangeModel.checked,
    learningStatus: selfLearning.status,
    learningScored: selfLearning.scored_rounds,
    learningRemaining: selfLearning.rounds_until_auto_select,
    learningBest: bestLearningModel.candidate_model,
    learningBestAccuracy: bestLearningModel.accuracy,
    mlAvailable: mlPrediction.available,
    mlDataUsed: mlPrediction.data_used_rounds,
    mlCurrent: mlPrediction.is_current,
    mlModelStatus: mlStatusLabel(mlPrediction),
    mlPredictions: mlEntries.map((item) => [
      item.target,
      item.probability,
      item.historical_baseline,
      item.edge,
      item.holdout_status,
    ]),
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
