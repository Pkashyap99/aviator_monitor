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
  signalQualityPanel: document.getElementById("signalQualityPanel"),
  signalQualityStatus: document.getElementById("signalQualityStatus"),
  signalQualityMain: document.getElementById("signalQualityMain"),
  signalQualityReasons: document.getElementById("signalQualityReasons"),
  dataQualityPanel: document.getElementById("dataQualityPanel"),
  dataQualityStatus: document.getElementById("dataQualityStatus"),
  dataQualityMain: document.getElementById("dataQualityMain"),
  dataQualityIssues: document.getElementById("dataQualityIssues"),
  sequenceWatchPanel: document.getElementById("sequenceWatchPanel"),
  sequenceWatchStatus: document.getElementById("sequenceWatchStatus"),
  sequenceWatchMain: document.getElementById("sequenceWatchMain"),
  sequenceWatchList: document.getElementById("sequenceWatchList"),
  aiWatchPanel: document.getElementById("aiWatchPanel"),
  aiWatchStatus: document.getElementById("aiWatchStatus"),
  aiWatchMain: document.getElementById("aiWatchMain"),
  aiWatchList: document.getElementById("aiWatchList"),
  accuracySummaryText: document.getElementById("accuracySummaryText"),
  selfLearningText: document.getElementById("selfLearningText"),
  sourceModeText: document.getElementById("sourceModeText"),
  collectorStatusText: document.getElementById("collectorStatusText"),
  roundContextText: document.getElementById("roundContextText"),
  bigRoundPanel: document.getElementById("bigRoundPanel"),
  bigRoundStatus: document.getElementById("bigRoundStatus"),
  bigRoundList: document.getElementById("bigRoundList"),
  timingPanel: document.getElementById("timingPanel"),
  timingStatus: document.getElementById("timingStatus"),
  timingSummary: document.getElementById("timingSummary"),
  timingList: document.getElementById("timingList"),
  mlPredictionPanel: document.getElementById("mlPredictionPanel"),
  mlModelStatus: document.getElementById("mlModelStatus"),
  mlProbabilityList: document.getElementById("mlProbabilityList"),
  mlModelMeta: document.getElementById("mlModelMeta"),
  modelHealthPanel: document.getElementById("modelHealthPanel"),
  modelHealthStatus: document.getElementById("modelHealthStatus"),
  modelHealthSummary: document.getElementById("modelHealthSummary"),
  modelHealthMetrics: document.getElementById("modelHealthMetrics"),
  modelHealthBigList: document.getElementById("modelHealthBigList"),
  strategyAuditPanel: document.getElementById("strategyAuditPanel"),
  strategyAuditStatus: document.getElementById("strategyAuditStatus"),
  strategyAuditSummary: document.getElementById("strategyAuditSummary"),
  strategyAuditList: document.getElementById("strategyAuditList"),
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

  const strongBig = bigPredictions.find(
    (prediction) => (
      prediction.clear_signal
      && prediction.predicted_high
      && String(prediction.confidence || "").toLowerCase() === "high"
      && Number(prediction.edge || 0) >= 0.03
    )
  );

  if (strongBig) {
    return `Big round alert: ${Number(strongBig.target).toFixed(0)}x+ stronger than normal`;
  }

  const strongest = bigPredictions
    .slice()
    .sort((left, right) => Number(right.probability) - Number(left.probability))[0];

  if (!strongest) {
    return "Big round alert: waiting";
  }

  return `Big round alert: no confirmed signal (${formatPercent(strongest.probability)} for ${Number(strongest.target).toFixed(0)}x+)`;
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
      text: "No proven edge right now",
      meta: `History rates only: ${probabilityLine}`,
      signal: "Model status: same as history",
      cashout: "No reliable cashout target",
      bigWatch: "Big round alert: no confirmed signal",
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

function buildSelectiveFastDisplay(signalQuality) {
  const call = signalQuality && signalQuality.selective_call
    ? signalQuality.selective_call
    : null;

  if (!call || String(call.status || "").toLowerCase() === "no_call") {
    return null;
  }

  const status = String(call.status || "").toLowerCase();
  const probabilityText = call.probability !== null && call.probability !== undefined
    ? `Chance ${formatPercent(call.probability)}`
    : null;
  const historyText = call.baseline_probability !== null && call.baseline_probability !== undefined
    ? `history ${formatPercent(call.baseline_probability)}`
    : null;
  const accuracyText = call.accuracy !== null && call.accuracy !== undefined
    ? status === "defensive"
      ? `recent low-rate ${formatPercent(call.accuracy)}`
      : `recent check ${formatPercent(call.accuracy)}`
    : null;
  const checkedText = call.checked
    ? `${formatCount(call.checked)} scored`
    : null;
  const meta = [
    call.headline || signalQuality.headline,
    probabilityText,
    historyText,
    accuracyText,
    checkedText,
  ].filter(Boolean).join(" | ");

  if (status === "active") {
    return {
      tone: "fast-high",
      text: call.main_call || "Active call",
      meta: meta || "Validated edge signal",
      signal: "Signal: ACTIVE",
      cashout: call.cashout || "Use shown target only",
      bigWatch: "Big round alert: only if active signal says so",
    };
  }

  if (status === "defensive") {
    return {
      tone: "fast-range",
      text: call.main_call || "Defensive only",
      meta: meta || "Common outcome read",
      signal: "Signal: DEFENSIVE ONLY",
      cashout: call.cashout || "No high chase",
      bigWatch: "Big round alert: no confirmed signal",
    };
  }

  return null;
}

function renderMlPrediction(mlPrediction, mlRetrain) {
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
    elements.mlModelMeta.textContent = `Run ml_train.py once if model files are missing. ${mlRetrainText(mlRetrain)} | ${mlLiveMetricText(mlRetrain)}`;
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
  elements.mlModelMeta.textContent = `${roundsText} | ${modelNames.join(", ") || "model"} | ${currentText} | ${mlRetrainText(mlRetrain)} | ${mlLiveMetricText(mlRetrain)}`;
}

function mlRetrainText(mlRetrain) {
  if (!mlRetrain) {
    return "Auto learning: checking";
  }

  if (!mlRetrain.enabled) {
    return "Auto learning: off";
  }

  if (mlRetrain.status === "training") {
    return "Auto learning: retraining now";
  }

  if (mlRetrain.status === "failed") {
    return "Auto learning: last retrain failed";
  }

  if (mlRetrain.status === "complete" && mlRetrain.last_success_at) {
    const remaining = Number(mlRetrain.rounds_until_next_train || 0);
    return remaining > 0
      ? `Auto learning: refreshed at ${mlRetrain.last_success_at}, next in ${formatCount(remaining)} rounds`
      : `Auto learning: refreshed at ${mlRetrain.last_success_at}`;
  }

  const remaining = Number(mlRetrain.rounds_until_next_train || 0);

  if (remaining <= 0) {
    return "Auto learning: retrain due soon";
  }

  return `Auto learning: next retrain in ${formatCount(remaining)} rounds`;
}

function mlLiveMetricText(mlRetrain) {
  const liveMetrics = mlRetrain && mlRetrain.live_metrics
    ? mlRetrain.live_metrics
    : {};
  const metric = (
    liveMetrics["2.00"] && (
      liveMetrics["2.00"]["100"]
      || liveMetrics["2.00"].all
    )
  ) || null;

  if (!metric || !metric.predictions) {
    return "Live score: collecting";
  }

  const skill = metric.brier_skill === null || metric.brier_skill === undefined
    ? "--"
    : formatPercentagePoints(metric.brier_skill);
  return `Live 2x score: ${formatPercent(metric.accuracy)} over ${formatCount(metric.predictions)} checks, skill ${skill}`;
}

function liveMetricForTarget(mlRetrain, target) {
  const liveMetrics = mlRetrain && mlRetrain.live_metrics
    ? mlRetrain.live_metrics
    : {};
  const key = Number(target).toFixed(2);
  const targetMetrics = liveMetrics[key] || {};
  return targetMetrics["500"]
    || targetMetrics["250"]
    || targetMetrics["100"]
    || targetMetrics.all
    || null;
}

function modelHealthStatusText(mlPrediction, mlRetrain) {
  if (mlRetrain && mlRetrain.status === "training") {
    return "Testing new model now";
  }

  const promotedTargets = mlRetrain && Array.isArray(mlRetrain.promoted_targets)
    ? mlRetrain.promoted_targets
    : [];

  if (promotedTargets.length) {
    return "Improved model active";
  }

  if (mlHasProvenEdge(mlPrediction)) {
    return "Model edge detected";
  }

  return "No proven edge yet";
}

function readableModelName(name) {
  if (!name) {
    return "history baseline";
  }

  return String(name).replaceAll("_", " ");
}

function championModelText(mlRetrain) {
  const targets = mlRetrain && mlRetrain.champion_targets
    ? mlRetrain.champion_targets
    : {};
  const names = [...new Set(Object.values(targets)
    .map((item) => readableModelName(item && item.model_name))
    .filter(Boolean))];

  if (!names.length) {
    return "history baseline";
  }

  if (names.length === 1) {
    return names[0];
  }

  return `${names.length} models`;
}

function simpleRejectReason(mlRetrain) {
  const kept = mlRetrain && Array.isArray(mlRetrain.kept_targets)
    ? mlRetrain.kept_targets
    : [];

  if (!kept.length) {
    return "";
  }

  const meaningful = kept.find((item) => item.reason !== "candidate is already champion model type")
    || kept[0];
  const reason = String(meaningful.reason || "");

  if (reason.includes("too small")) {
    return "improvement was too small";
  }

  if (reason.includes("holdout")) {
    return "latest unseen rounds were weak";
  }

  if (reason.includes("calibration")) {
    return "probability quality was weak";
  }

  return "new model did not beat history enough";
}

function lastRetrainText(mlRetrain) {
  if (!mlRetrain) {
    return "Last retrain: checking";
  }

  if (mlRetrain.status === "training") {
    return "Last retrain: running now";
  }

  if (mlRetrain.status === "failed") {
    return "Last retrain: failed";
  }

  const promoted = Array.isArray(mlRetrain.promoted_targets)
    ? mlRetrain.promoted_targets
    : [];

  if (promoted.length) {
    return `Last retrain: improved ${promoted.join(", ")}`;
  }

  const kept = Array.isArray(mlRetrain.kept_targets)
    ? mlRetrain.kept_targets
    : [];

  if (kept.length) {
    const reason = simpleRejectReason(mlRetrain);
    return reason
      ? `Last retrain: no promotion, ${reason}`
      : "Last retrain: no promotion";
  }

  if (mlRetrain.last_success_at) {
    return `Last retrain: checked at ${mlRetrain.last_success_at}`;
  }

  return "Last retrain: waiting";
}

function nextRetrainText(mlRetrain) {
  if (!mlRetrain) {
    return "checking";
  }

  if (!mlRetrain.enabled) {
    return "off";
  }

  if (mlRetrain.status === "training") {
    return "now";
  }

  const remaining = Number(mlRetrain.rounds_until_next_train || 0);
  return remaining > 0
    ? `${formatCount(remaining)} rounds`
    : "due soon";
}

function liveBaselineText(metric) {
  if (!metric || !metric.predictions) {
    return "collecting live checks";
  }

  const skill = Number(metric.brier_skill || 0);

  if (skill > 0.005) {
    return `beating baseline by ${formatPercentagePoints(skill)}`;
  }

  if (skill < -0.005) {
    return `below baseline by ${formatPercentagePoints(skill)}`;
  }

  return "same as baseline";
}

function appendModelHealthMetric(parent, label, value, note, tone = "") {
  const card = document.createElement("div");
  card.className = ["model-health-metric", tone].filter(Boolean).join(" ");

  const labelNode = document.createElement("span");
  labelNode.textContent = label;
  const valueNode = document.createElement("strong");
  valueNode.textContent = value;
  const noteNode = document.createElement("small");
  noteNode.textContent = note;

  card.append(labelNode, valueNode, noteNode);
  parent.appendChild(card);
}

function renderModelHealthBigList(bigRounds) {
  if (!elements.modelHealthBigList) {
    return;
  }

  elements.modelHealthBigList.innerHTML = "";
  const thresholds = Array.isArray(bigRounds && bigRounds.thresholds)
    ? bigRounds.thresholds
    : [];
  const wanted = [10, 20, 50, 100]
    .map((target) => thresholds.find((item) => Number(item.target) === target))
    .filter(Boolean);

  if (!wanted.length) {
    elements.modelHealthBigList.textContent = "Big multiplier history: collecting";
    return;
  }

  for (const item of wanted) {
    const target = Number(item.target);
    const row = document.createElement("div");
    row.className = "model-health-big-row";

    const label = document.createElement("span");
    label.textContent = `${target.toFixed(0)}x+`;
    const rate = document.createElement("strong");
    rate.textContent = formatPercent(item.rate);
    const note = document.createElement("small");
    note.textContent = bigRoundGapText(item);

    row.append(label, rate, note);
    elements.modelHealthBigList.appendChild(row);
  }
}

function renderModelHealth(data) {
  if (
    !elements.modelHealthPanel
    || !elements.modelHealthStatus
    || !elements.modelHealthSummary
    || !elements.modelHealthMetrics
  ) {
    return;
  }

  const mlRetrain = data.ml_retrain || null;
  const mlPrediction = data.ml_prediction || null;
  const hasEdge = mlHasProvenEdge(mlPrediction);
  const statusText = modelHealthStatusText(mlPrediction, mlRetrain);
  const live2x = liveMetricForTarget(mlRetrain, 2);
  const currentRounds = mlRetrain && mlRetrain.current_rounds
    ? formatCount(mlRetrain.current_rounds)
    : formatCount((data.summary || {}).rounds);
  const newRounds = mlRetrain && mlRetrain.new_rounds_since_train !== undefined
    ? formatCount(mlRetrain.new_rounds_since_train)
    : "--";

  elements.modelHealthPanel.classList.toggle("has-edge", hasEdge);
  elements.modelHealthPanel.classList.toggle(
    "is-training",
    Boolean(mlRetrain && mlRetrain.status === "training"),
  );
  elements.modelHealthStatus.textContent = statusText;
  elements.modelHealthSummary.textContent = lastRetrainText(mlRetrain);
  elements.modelHealthMetrics.innerHTML = "";

  appendModelHealthMetric(
    elements.modelHealthMetrics,
    "Current model",
    championModelText(mlRetrain),
    hasEdge ? "using promoted signal" : "following history baseline",
    hasEdge ? "good" : "",
  );
  appendModelHealthMetric(
    elements.modelHealthMetrics,
    "Next test",
    nextRetrainText(mlRetrain),
    "automatic challenger check",
  );
  appendModelHealthMetric(
    elements.modelHealthMetrics,
    "Live 2x score",
    live2x && live2x.predictions
      ? `${formatPercent(live2x.accuracy)}`
      : "--",
    liveBaselineText(live2x),
    live2x && Number(live2x.brier_skill || 0) > 0.005 ? "good" : "",
  );
  appendModelHealthMetric(
    elements.modelHealthMetrics,
    "Data used",
    `${currentRounds} rounds`,
    `${newRounds} new since last model test`,
  );

  renderModelHealthBigList(data.big_rounds || {});
}

function roiText(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "--";
  }

  const number = Number(value);
  const sign = number > 0 ? "+" : "";
  return `${sign}${formatPercent(number)}`;
}

function appendStrategyAuditRow(parent, label, item) {
  if (!item) {
    return;
  }

  const row = document.createElement("div");
  row.className = "strategy-audit-row";

  const title = document.createElement("span");
  title.textContent = label;

  const strategy = document.createElement("strong");
  strategy.textContent = item.label || "strategy";

  const detail = document.createElement("small");
  const train = item.train || {};
  const holdout = item.holdout || {};
  detail.textContent = [
    `old ${roiText(train.roi)} (${formatCount(train.bets || 0)} bets)`,
    `later ${roiText(holdout.roi)} (${formatCount(holdout.bets || 0)} bets)`,
  ].join(" | ");

  row.append(title, strategy, detail);
  parent.appendChild(row);
}

function renderStrategyAudit(strategyAudit) {
  if (
    !elements.strategyAuditPanel
    || !elements.strategyAuditStatus
    || !elements.strategyAuditSummary
    || !elements.strategyAuditList
  ) {
    return;
  }

  const audit = strategyAudit || {};
  const status = String(audit.status || "missing").toLowerCase();

  elements.strategyAuditPanel.classList.remove("candidate", "no-edge", "stale", "missing");
  elements.strategyAuditPanel.classList.add(status.replace("_", "-"));
  elements.strategyAuditStatus.textContent = audit.headline || "Strategy audit missing";

  if (!audit.available) {
    elements.strategyAuditSummary.textContent = audit.message || "Run strategy_audit.py to test saved rounds.";
    elements.strategyAuditList.textContent = "";
    return;
  }

  const staleText = audit.stale
    ? ` | stale by ${formatCount(audit.new_rounds || 0)} rounds`
    : "";
  elements.strategyAuditSummary.textContent = [
    audit.message || "Strategy audit ready.",
    `${formatCount(audit.train_rounds || 0)} old / ${formatCount(audit.holdout_rounds || 0)} later rounds${staleText}`,
  ].join(" ");
  elements.strategyAuditList.innerHTML = "";

  appendStrategyAuditRow(
    elements.strategyAuditList,
    "Best old",
    audit.best_train_strategy,
  );
  appendStrategyAuditRow(
    elements.strategyAuditList,
    "Forward watch",
    audit.best_forward_candidate,
  );

  if (!audit.best_forward_candidate) {
    const note = document.createElement("small");
    note.className = "strategy-audit-note";
    note.textContent = "No strategy stayed positive on later holdout.";
    elements.strategyAuditList.appendChild(note);
  }
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

function renderTimingInsights(timing) {
  if (!elements.timingPanel || !elements.timingStatus || !elements.timingSummary || !elements.timingList) {
    return;
  }

  elements.timingPanel.classList.remove("has-window");
  elements.timingList.innerHTML = "";

  if (!timing || !timing.available) {
    elements.timingStatus.textContent = "Collecting timing";
    elements.timingSummary.textContent = timing && timing.minimum_rounds
      ? `${formatCount(timing.rounds)} rounds saved, needs ${formatCount(timing.minimum_rounds)}`
      : "Historical timing only";
    return;
  }

  const topWindows = Array.isArray(timing.top_windows) ? timing.top_windows : [];
  const best = topWindows[0] || null;
  const sameTime = timing.same_time_last_week || {};
  const sameTimeWindows = sameTime && Array.isArray(sameTime.windows)
    ? sameTime.windows
    : [];

  elements.timingStatus.textContent = best && Number(best.score) > 0
    ? best.label
    : "No clear time edge";
  elements.timingSummary.textContent = sameTime.available
    ? `Latest vs same time last week | ${formatCount(timing.rounds)} rounds checked`
    : `${formatCount(timing.rounds)} rounds checked | ${formatCount(timing.minimum_bucket_rounds)}+ per window`;
  elements.timingPanel.classList.toggle("has-window", Boolean(best && Number(best.score) > 0));

  if (sameTime.available && sameTimeWindows.length) {
    for (const item of sameTimeWindows.slice(0, 3)) {
      const current = item.current || {};
      const lastWeek = item.last_week || {};
      const currentRates = current.rates || {};
      const lastWeekRates = lastWeek.rates || {};
      const deltas = item.deltas || {};
      const row = document.createElement("div");
      row.className = "timing-row timing-compare-row";

      const label = document.createElement("span");
      label.className = "timing-label";
      label.textContent = `${item.label} vs last week`;

      const ratesGrid = document.createElement("div");
      ratesGrid.className = "timing-rate-grid";

      for (const target of ["2.00", "5.00", "10.00"]) {
        const cell = document.createElement("div");
        cell.className = "timing-rate-cell";

        const targetLabel = document.createElement("em");
        targetLabel.textContent = `${Number(target).toFixed(0)}x+`;

        const value = document.createElement("strong");
        value.textContent = `${formatPercent(currentRates[target])} vs ${formatPercent(lastWeekRates[target])}`;

        const delta = document.createElement("small");
        delta.textContent = `change ${formatPercentagePoints(deltas[target])}`;

        cell.appendChild(targetLabel);
        cell.appendChild(value);
        cell.appendChild(delta);
        ratesGrid.appendChild(cell);
      }

      const detail = document.createElement("small");
      detail.textContent = [
        `now ${formatCount(current.rounds)} rounds`,
        `last week ${formatCount(lastWeek.rounds)} rounds`,
        sameTime.last_week_timestamp ? `reference ${sameTime.last_week_timestamp}` : null,
      ].filter(Boolean).join(" | ");

      row.appendChild(label);
      row.appendChild(ratesGrid);
      row.appendChild(detail);
      elements.timingList.appendChild(row);
    }
  }

  const rows = topWindows.slice(0, 4);

  if (!rows.length) {
    if (!elements.timingList.children.length) {
      elements.timingList.textContent = "No weekday/hour bucket has enough data yet.";
    }
    return;
  }

  for (const item of rows) {
    const rates = item.rates || {};
    const row = document.createElement("div");
    row.className = "timing-row";

    const label = document.createElement("span");
    label.className = "timing-label";
    label.textContent = item.label || "--";

    const ratesGrid = document.createElement("div");
    ratesGrid.className = "timing-rate-grid";

    for (const target of ["2.00", "5.00", "10.00"]) {
      const cell = document.createElement("div");
      cell.className = "timing-rate-cell";

      const targetLabel = document.createElement("em");
      targetLabel.textContent = `${Number(target).toFixed(0)}x+`;

      const value = document.createElement("strong");
      value.textContent = formatPercent(rates[target]);

      cell.appendChild(targetLabel);
      cell.appendChild(value);
      ratesGrid.appendChild(cell);
    }

    const detail = document.createElement("small");
    detail.textContent = [
      `${formatCount(item.rounds)} rounds`,
      `edge ${formatPercentagePoints(item.score)}`,
    ].join(" | ");

    row.appendChild(label);
    row.appendChild(ratesGrid);
    row.appendChild(detail);
    elements.timingList.appendChild(row);
  }

  const note = document.createElement("small");
  note.className = "timing-note";
  note.textContent = timing.note || "Historical timing only; not a guarantee.";
  elements.timingList.appendChild(note);
}

function renderSignalQuality(quality) {
  if (
    !elements.signalQualityPanel
    || !elements.signalQualityStatus
    || !elements.signalQualityMain
    || !elements.signalQualityReasons
  ) {
    return;
  }

  const status = String((quality && quality.status) || "wait").toLowerCase();
  const reasons = quality && Array.isArray(quality.reasons) ? quality.reasons : [];

  elements.signalQualityPanel.classList.remove("wait", "watch", "active");
  elements.signalQualityPanel.classList.add(status);
  elements.signalQualityStatus.textContent = quality && quality.headline
    ? quality.headline
    : "Wait - no proven edge";

  const scoreText = quality && Number.isFinite(Number(quality.score))
    ? `Quality ${Math.round(Number(quality.score))}/100`
    : "Quality --";
  const mainCall = quality && quality.main_call ? quality.main_call : "No play signal";
  const cashout = quality && quality.cashout ? quality.cashout : "No reliable cashout target.";
  elements.signalQualityMain.textContent = `${scoreText} | ${mainCall} | ${cashout}`;
  elements.signalQualityReasons.innerHTML = "";

  for (const reason of reasons.slice(0, 4)) {
    const row = document.createElement("div");
    row.className = `signal-reason ${reason.tone || "neutral"}`;

    const label = document.createElement("span");
    label.textContent = reason.label || "Check";

    const detail = document.createElement("small");
    detail.textContent = reason.detail || "--";

    row.appendChild(label);
    row.appendChild(detail);
    elements.signalQualityReasons.appendChild(row);
  }
}

function renderDataQuality(quality) {
  if (
    !elements.dataQualityPanel
    || !elements.dataQualityStatus
    || !elements.dataQualityMain
    || !elements.dataQualityIssues
  ) {
    return;
  }

  elements.dataQualityPanel.classList.remove("good", "watch", "bad", "empty");
  elements.dataQualityIssues.innerHTML = "";

  if (!quality || !quality.available) {
    elements.dataQualityPanel.classList.add("empty");
    elements.dataQualityStatus.textContent = "No data yet";
    elements.dataQualityMain.textContent = "Waiting for saved rounds.";
    return;
  }

  const status = String(quality.status || "watch").toLowerCase();
  const normalizedStatus = ["good", "watch", "bad"].includes(status)
    ? status
    : "watch";
  const score = Number(quality.score);
  const scoreText = Number.isFinite(score) ? `${Math.round(score)}/100` : "--";
  const intervals = quality.intervals || {};
  const recentIntervals = quality.recent_intervals || {};
  const usualGap = intervals.median_seconds !== null && intervals.median_seconds !== undefined
    ? `${Number(intervals.median_seconds).toFixed(1)}s usual gap`
    : "usual gap unknown";
  const recentGaps = Number(recentIntervals.possible_capture_gap_count || 0);
  const gapText = recentGaps > 0
    ? `${recentGaps} recent gap${recentGaps === 1 ? "" : "s"}`
    : "no recent gaps";
  const ageText = quality.last_round_age_seconds !== null && quality.last_round_age_seconds !== undefined
    ? `last ${formatAge(quality.last_round_age_seconds)} ago`
    : "last time unknown";

  elements.dataQualityPanel.classList.add(normalizedStatus);
  elements.dataQualityStatus.textContent = `${quality.headline || "Data check"} (${scoreText})`;
  elements.dataQualityMain.textContent = [
    `${formatCount(quality.valid_rows)} usable rounds`,
    usualGap,
    gapText,
    ageText,
  ].join(" | ");

  const issues = Array.isArray(quality.issues) ? quality.issues : [];

  for (const issue of issues.slice(0, 3)) {
    const row = document.createElement("div");
    row.className = `data-quality-issue ${issue.severity || "info"}`;

    const label = document.createElement("span");
    label.textContent = issue.label || "Check";

    const detail = document.createElement("small");
    detail.textContent = issue.detail || "--";

    row.appendChild(label);
    row.appendChild(detail);
    elements.dataQualityIssues.appendChild(row);
  }
}

function plainSequencePattern(pattern) {
  return String(pattern || "")
    .replace(/^last\s+\d+\s+sequence:\s*/i, "")
    .replace(/\s+->\s+/g, " then ");
}

function plainOutcomeText(outcome) {
  const text = String(outcome || "");
  const higher = text.match(/^next\s+>=\s+(.+)$/i);

  if (higher) {
    return `${higher[1]} or higher`;
  }

  const lower = text.match(/^next\s+<\s+(.+)$/i);

  if (lower) {
    return `below ${lower[1]}`;
  }

  return text || "--";
}

function sequenceWatchChanceText(item) {
  if (!item) {
    return "--";
  }

  const rate = formatPercent(item.holdout_rate);
  const baseline = formatPercent(item.holdout_baseline);
  const lift = Number(item.holdout_lift || 0);
  const liftText = lift > 0
    ? `${formatPercentagePoints(lift)} better`
    : formatPercentagePoints(lift);
  const checked = Number(item.holdout_checked);
  const checkedText = Number.isFinite(checked)
    ? `${formatCount(checked)} past matches`
    : "past matches unavailable";

  return `Past chance ${rate}. Normal chance ${baseline}. ${liftText}. Checked ${checkedText}.`;
}

function renderSequenceWatch(sequenceWatch) {
  if (
    !elements.sequenceWatchPanel
    || !elements.sequenceWatchStatus
    || !elements.sequenceWatchMain
    || !elements.sequenceWatchList
  ) {
    return;
  }

  elements.sequenceWatchPanel.classList.remove("active", "watch", "empty");
  elements.sequenceWatchList.innerHTML = "";

  const currentSequences = sequenceWatch && Array.isArray(sequenceWatch.current_sequences)
    ? sequenceWatch.current_sequences
    : [];
  const currentShort = currentSequences.length
    ? currentSequences[0].label
    : "waiting for last rounds";

  if (!sequenceWatch || !sequenceWatch.available) {
    elements.sequenceWatchPanel.classList.add("empty");
    elements.sequenceWatchStatus.textContent = "Checking";
    elements.sequenceWatchMain.textContent = sequenceWatch && sequenceWatch.message
      ? sequenceWatch.message
      : `Current: ${currentShort}`;
    return;
  }

  const active = Array.isArray(sequenceWatch.active) ? sequenceWatch.active : [];
  const weakActive = Array.isArray(sequenceWatch.weak_active) ? sequenceWatch.weak_active : [];
  const confirmed = Array.isArray(sequenceWatch.confirmed) ? sequenceWatch.confirmed : [];
  const watch = Array.isArray(sequenceWatch.watch) ? sequenceWatch.watch : [];
  const bestActive = active[0] || null;
  const bestWeakActive = weakActive[0] || null;
  const bestConfirmed = confirmed[0] || null;

  if (bestActive) {
    elements.sequenceWatchPanel.classList.add("active");
    elements.sequenceWatchStatus.textContent = "Strong big-round alert";
    elements.sequenceWatchMain.textContent = `Recent rounds match a confirmed pattern. Target to watch: ${plainOutcomeText(bestActive.outcome)}. Still not guaranteed.`;
  } else if (bestWeakActive) {
    elements.sequenceWatchPanel.classList.add("watch");
    elements.sequenceWatchStatus.textContent = "Weak match, no alert";
    elements.sequenceWatchMain.textContent = `Recent rounds match a weak pattern for ${plainOutcomeText(bestWeakActive.outcome)}, but it is not reliable enough for a big-round alert.`;
  } else if (bestConfirmed) {
    elements.sequenceWatchPanel.classList.add("watch");
    elements.sequenceWatchStatus.textContent = "No big signal now";
    elements.sequenceWatchMain.textContent = `Current pattern: ${currentShort}. Waiting for: ${plainSequencePattern(bestConfirmed.pattern)}.`;
  } else if (watch.length) {
    elements.sequenceWatchPanel.classList.add("watch");
    elements.sequenceWatchStatus.textContent = "Weak watch only";
    elements.sequenceWatchMain.textContent = `Current pattern: ${currentShort}. No confirmed big-round signal yet.`;
  } else {
    elements.sequenceWatchPanel.classList.add("empty");
    elements.sequenceWatchStatus.textContent = "No clear pattern";
    elements.sequenceWatchMain.textContent = `Current pattern: ${currentShort}. No useful big-round signal found.`;
  }

  const rows = active.length
    ? active
    : weakActive.length
      ? weakActive
      : confirmed.concat(watch).slice(0, 3);

  for (const item of rows) {
    const row = document.createElement("div");
    const isStrongActive = item.active && item.status === "confirmed";
    row.className = `sequence-watch-row ${isStrongActive ? "active" : ""} ${item.status || ""}`;

    const label = document.createElement("span");
    label.textContent = isStrongActive
      ? `Strong alert: ${plainOutcomeText(item.outcome)}`
      : item.active
        ? `Weak match only: ${plainOutcomeText(item.outcome)}`
      : `Wait for: ${plainSequencePattern(item.pattern)} | Target: ${plainOutcomeText(item.outcome)}`;

    const detail = document.createElement("small");
    detail.textContent = sequenceWatchChanceText(item);

    row.appendChild(label);
    row.appendChild(detail);
    elements.sequenceWatchList.appendChild(row);
  }
}

function edgeAuditRateText(item) {
  if (!item) {
    return "--";
  }

  const rate = formatPercent(item.holdout_rate);
  const baseline = formatPercent(item.holdout_baseline);
  const lift = Number(item.holdout_lift || 0);
  const liftText = lift > 0
    ? `+${formatPercentagePoints(lift)}`
    : formatPercentagePoints(lift);
  const qValue = Number(item.holdout_q_value);
  const qText = Number.isFinite(qValue)
    ? `q ${qValue.toFixed(3)}`
    : "q n/a";
  const ciLow = Number(item.holdout_lift_ci_low);
  const ciHigh = Number(item.holdout_lift_ci_high);
  const ciText = Number.isFinite(ciLow) && Number.isFinite(ciHigh)
    ? `CI ${formatPercentagePoints(ciLow)}..${formatPercentagePoints(ciHigh)}`
    : "CI n/a";
  const positiveFolds = Number(item.walk_forward_positive_folds);
  const validFolds = Number(item.walk_forward_valid_folds);
  const significantFolds = Number(item.walk_forward_significant_positive_folds);
  const walkText = Number.isFinite(positiveFolds) && Number.isFinite(validFolds)
    ? `WF ${positiveFolds}/${validFolds}+${Number.isFinite(significantFolds) ? `, sig ${significantFolds}` : ""}`
    : "WF n/a";

  return `${rate} vs ${baseline} (${liftText}, ${qText}, ${walkText}, ${ciText})`;
}

function renderAiWatch(edgeAudit) {
  if (
    !elements.aiWatchPanel
    || !elements.aiWatchStatus
    || !elements.aiWatchMain
    || !elements.aiWatchList
  ) {
    return;
  }

  elements.aiWatchPanel.classList.remove("active", "watch", "empty");
  elements.aiWatchList.innerHTML = "";

  if (!edgeAudit || !edgeAudit.available) {
    elements.aiWatchPanel.classList.add("empty");
    elements.aiWatchStatus.textContent = "No audit yet";
    elements.aiWatchMain.textContent = edgeAudit && edgeAudit.message
      ? edgeAudit.message
      : "Run edge audit to search for watch patterns.";
    return;
  }

  const active = Array.isArray(edgeAudit.active) ? edgeAudit.active : [];
  const watch = Array.isArray(edgeAudit.watch) ? edgeAudit.watch : [];
  const bestActive = active[0] || null;
  const hasConfirmed = Number(edgeAudit.fdr_confirmed_count || 0) > 0;
  const hasStable = Number(edgeAudit.walk_forward_stable_count || 0) > 0;
  const hasRawWatch = Number(edgeAudit.watch_candidate_count || 0) > 0;
  const testedText = edgeAudit.patterns_tested
    ? `${edgeAudit.patterns_tested} patterns tested`
    : "Patterns tested";
  const foldText = edgeAudit.walk_forward_folds
    ? `${edgeAudit.walk_forward_folds} time windows`
    : "walk-forward windows";

  if (bestActive) {
    elements.aiWatchPanel.classList.add("active");
    elements.aiWatchStatus.textContent = bestActive.walk_forward_stable || bestActive.strong_edge
      ? "Stable watch active"
      : bestActive.fdr_confirmed
        ? "Confirmed watch active"
        : "Unconfirmed watch active";
    elements.aiWatchMain.textContent = `${bestActive.condition} -> ${bestActive.target}x+`;
  } else if (hasStable) {
    elements.aiWatchPanel.classList.add("watch");
    elements.aiWatchStatus.textContent = "Stable watch found";
    elements.aiWatchMain.textContent = `No stable pattern active now. ${testedText}, ${foldText}.`;
  } else if (hasConfirmed) {
    elements.aiWatchPanel.classList.add("watch");
    elements.aiWatchStatus.textContent = "Confirmed but unstable";
    elements.aiWatchMain.textContent = `No confirmed pattern active now. ${testedText}, ${foldText}.`;
  } else if (hasRawWatch) {
    elements.aiWatchPanel.classList.add("watch");
    elements.aiWatchStatus.textContent = "No proven edge";
    elements.aiWatchMain.textContent = `Weak hints only. They did not pass confirmation across ${foldText}.`;
  } else {
    elements.aiWatchPanel.classList.add("empty");
    elements.aiWatchStatus.textContent = "No proven edge";
    elements.aiWatchMain.textContent = edgeAudit.conclusion || `No repeatable watch pattern found. ${testedText}.`;
  }

  const rows = active.length ? active : watch.slice(0, 3);

  for (const item of rows) {
    const row = document.createElement("div");
    row.className = `ai-watch-row ${item.active ? "active" : ""}`;

    const label = document.createElement("span");
    label.textContent = `${item.condition} -> ${item.target}x+`;

    const detail = document.createElement("small");
    detail.textContent = edgeAuditRateText(item);

    row.appendChild(label);
    row.appendChild(detail);
    elements.aiWatchList.appendChild(row);
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

function renderCollectorStatus(status) {
  if (!elements.collectorStatusText) {
    return;
  }

  if (!status || !status.available) {
    elements.collectorStatusText.textContent = "Game: waiting for live state";
    elements.collectorStatusText.className = "collector-status";
    return;
  }

  const phase = String(status.phase || "");
  const label = status.label || "Game state detected";
  const multiplier = status.live_multiplier !== null && status.live_multiplier !== undefined
    ? ` ${formatMultiplier(status.live_multiplier)}`
    : "";
  const age = status.age_seconds !== null && status.age_seconds !== undefined
    ? ` - ${formatAge(status.age_seconds)} ago`
    : "";
  const realtime = status.realtime_channels && status.realtime_channels.available
    ? " - realtime connected"
    : "";

  elements.collectorStatusText.textContent = `Game: ${label}${multiplier}${age}${realtime}`;
  elements.collectorStatusText.className = `collector-status ${phase || "unknown"}`;
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
    elements.bigWatchText.textContent = "Big round alert: waiting";
    elements.fastSignalText.textContent = liveDataLabel(data);
    elements.cashoutGuideText.textContent = liveDataDetail(data, "wait for live data");
    return;
  }

  const selectiveDisplay = buildSelectiveFastDisplay(data.signal_quality);

  if (selectiveDisplay) {
    elements.fastMain.classList.add(selectiveDisplay.tone);
    elements.fastPredictionText.textContent = selectiveDisplay.text;
    elements.fastPredictionMeta.textContent = selectiveDisplay.meta;
    elements.signalStrengthText.textContent = selectiveDisplay.signal;
    elements.bigWatchText.textContent = selectiveDisplay.bigWatch;
    elements.fastSignalText.textContent = liveDataLabel(data);
    elements.cashoutGuideText.textContent = liveDataDetail(data, selectiveDisplay.cashout);
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
    elements.bigWatchText.textContent = "Big round alert: waiting";
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
  renderCollectorStatus(data.collector_status);
  renderRoundContext(data.round_context);
  renderSignalQuality(data.signal_quality);
  renderDataQuality(data.data_quality);
  renderSequenceWatch(data.sequence_watch);
  renderAiWatch(data.edge_audit);
  renderBigRounds(data.big_rounds);
  renderTimingInsights(data.timing_insights);
  renderMlPrediction(data.ml_prediction, data.ml_retrain);
  renderModelHealth(data);
  renderStrategyAudit(data.strategy_audit);
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
  const mlRetrain = data.ml_retrain || {};
  const collectorStatus = data.collector_status || {};
  const timing = data.timing_insights || {};
  const signalQuality = data.signal_quality || {};
  const dataQuality = data.data_quality || {};
  const dataQualityIntervals = dataQuality.intervals || {};
  const dataQualityRecentIntervals = dataQuality.recent_intervals || {};
  const edgeAudit = data.edge_audit || {};
  const strategyAudit = data.strategy_audit || {};
  const bestStrategy = strategyAudit.best_train_strategy || {};
  const forwardStrategy = strategyAudit.best_forward_candidate || {};
  const sequenceWatch = data.sequence_watch || {};

  return JSON.stringify({
    rounds: summary.rounds,
    latest: summary.latest_multiplier,
    age: ingest.last_round_age_seconds,
    gamePhase: collectorStatus.phase,
    gameLiveMultiplier: collectorStatus.live_multiplier,
    gameStatusAge: collectorStatus.age_seconds,
    gameObservedAt: collectorStatus.observed_at,
    realtimeChannels: collectorStatus.realtime_channels
      ? [
          collectorStatus.realtime_channels.total,
          collectorStatus.realtime_channels.observed_at,
        ]
      : null,
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
    timingMessage: timing.message,
    timingWindows: (timing.top_windows || []).map((item) => [
      item.label,
      item.rounds,
      item.score,
      item.rates && item.rates["2.00"],
      item.rates && item.rates["5.00"],
      item.rates && item.rates["10.00"],
    ]),
    timingSameTimeLastWeek: timing.same_time_last_week
      ? [
          timing.same_time_last_week.available,
          timing.same_time_last_week.anchor_timestamp,
          timing.same_time_last_week.last_week_timestamp,
          (timing.same_time_last_week.windows || []).map((item) => [
            item.label,
            item.current && item.current.rounds,
            item.last_week && item.last_week.rounds,
            item.current && item.current.rates && item.current.rates["2.00"],
            item.last_week && item.last_week.rates && item.last_week.rates["2.00"],
            item.current && item.current.rates && item.current.rates["5.00"],
            item.last_week && item.last_week.rates && item.last_week.rates["5.00"],
            item.current && item.current.rates && item.current.rates["10.00"],
            item.last_week && item.last_week.rates && item.last_week.rates["10.00"],
          ]),
        ]
      : null,
    signalQualityStatus: signalQuality.status,
    signalQualityScore: signalQuality.score,
    signalQualityHeadline: signalQuality.headline,
    signalQualityReasons: (signalQuality.reasons || []).map((item) => [
      item.label,
      item.detail,
      item.tone,
    ]),
    dataQualityStatus: dataQuality.status,
    dataQualityScore: dataQuality.score,
    dataQualityValidRows: dataQuality.valid_rows,
    dataQualityDuplicates: dataQuality.duplicate_exact_count,
    dataQualityPossibleNoIdDuplicates: dataQuality.possible_no_id_duplicate_count,
    dataQualityRecentGaps: dataQualityRecentIntervals.possible_capture_gap_count,
    dataQualityOldGaps: dataQualityIntervals.possible_capture_gap_count,
    dataQualityMedianGap: dataQualityIntervals.median_seconds,
    dataQualityIssues: (dataQuality.issues || []).map((item) => [
      item.severity,
      item.label,
      item.detail,
    ]),
    sequenceWatchGeneratedAt: sequenceWatch.generated_at,
    sequenceWatchCurrent: (sequenceWatch.current_sequences || []).map((item) => [
      item.key,
      item.label,
    ]),
    sequenceWatchActive: (sequenceWatch.active || []).map((item) => [
      item.pattern_key,
      item.outcome_key,
      item.status,
      item.holdout_rate,
      item.holdout_lift,
      item.holdout_q_value,
      item.walk_forward_valid_folds,
      item.walk_forward_positive_folds,
      item.walk_forward_significant_positive_folds,
    ]),
    sequenceWatchWeakActive: (sequenceWatch.weak_active || []).map((item) => [
      item.pattern_key,
      item.outcome_key,
      item.status,
      item.holdout_rate,
      item.holdout_lift,
      item.holdout_q_value,
      item.walk_forward_valid_folds,
      item.walk_forward_positive_folds,
      item.walk_forward_significant_positive_folds,
    ]),
    sequenceWatchConfirmed: (sequenceWatch.confirmed || []).slice(0, 3).map((item) => [
      item.pattern_key,
      item.outcome_key,
      item.status,
      item.holdout_rate,
      item.holdout_lift,
      item.holdout_q_value,
      item.walk_forward_valid_folds,
      item.walk_forward_positive_folds,
      item.walk_forward_significant_positive_folds,
    ]),
    sequenceWatchWatch: (sequenceWatch.watch || []).slice(0, 3).map((item) => [
      item.pattern_key,
      item.outcome_key,
      item.status,
      item.holdout_rate,
      item.holdout_lift,
      item.holdout_q_value,
      item.walk_forward_valid_folds,
      item.walk_forward_positive_folds,
      item.walk_forward_significant_positive_folds,
    ]),
    edgeAuditGeneratedAt: edgeAudit.generated_at,
    edgeAuditActive: (edgeAudit.active || []).map((item) => [
      item.condition,
      item.target,
      item.holdout_rate,
      item.holdout_lift,
      item.holdout_q_value,
      item.holdout_lift_ci_low,
      item.holdout_lift_ci_high,
      item.fdr_confirmed,
      item.walk_forward_stable,
      item.walk_forward_valid_folds,
      item.walk_forward_positive_folds,
      item.walk_forward_significant_positive_folds,
    ]),
    edgeAuditWatch: (edgeAudit.watch || []).slice(0, 3).map((item) => [
      item.condition,
      item.target,
      item.holdout_rate,
      item.holdout_lift,
      item.holdout_q_value,
      item.holdout_lift_ci_low,
      item.holdout_lift_ci_high,
      item.fdr_confirmed,
      item.walk_forward_stable,
      item.walk_forward_valid_folds,
      item.walk_forward_positive_folds,
      item.walk_forward_significant_positive_folds,
    ]),
    strategyAuditStatus: strategyAudit.status,
    strategyAuditRounds: strategyAudit.rounds,
    strategyAuditNewRounds: strategyAudit.new_rounds,
    bestStrategy: [
      bestStrategy.label,
      bestStrategy.train && bestStrategy.train.roi,
      bestStrategy.holdout && bestStrategy.holdout.roi,
    ],
    forwardStrategy: [
      forwardStrategy.label,
      forwardStrategy.train && forwardStrategy.train.roi,
      forwardStrategy.holdout && forwardStrategy.holdout.roi,
    ],
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
    mlRetrainStatus: mlRetrain.status,
    mlRetrainCurrentRounds: mlRetrain.current_rounds,
    mlRetrainLastTrained: mlRetrain.last_trained_rounds,
    mlRetrainRemaining: mlRetrain.rounds_until_next_train,
    mlRetrainSuccessAt: mlRetrain.last_success_at,
    mlRetrainError: mlRetrain.last_error,
    mlRetrainMessage: mlRetrain.last_message,
    mlPromotedTargets: mlRetrain.promoted_targets || [],
    mlKeptTargets: (mlRetrain.kept_targets || []).map((item) => [
      item.target,
      item.candidate_model,
      item.reason,
    ]),
    mlLiveMetric: mlLiveMetricText(mlRetrain),
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
