"use strict";

const byId = (id) => document.getElementById(id);
const SVG_NS = "http://www.w3.org/2000/svg";
const BAND_COLORS = [
  "#286783", "#b77850", "#4d897d", "#8c6290", "#b99a48",
  "#4b739a", "#ad6871", "#65825d", "#947657", "#517c87"
];
const PLOT = {x0: 68, x1: 864, y0: 30, y1: 423};
const GAP_ANIMATION_MS = 650;
const state = {
  meta: null,
  design: [],
  example: [],
  revision: 0,
  lastEditAt: 0,
  kPoints: 30,
  device: "cpu",
  cudaReady: false,
  brush: 1,
  selected: [0, 15],
  undo: [],
  redo: [],
  drawing: false,
  lastPaintCell: null,
  strokeBase: null,
  strokeChanged: false,
  sliderBase: null,
  sliderChanged: false,
  prediction: null,
  fem: null,
  femBusy: false,
  predictBusy: false,
  predictPending: false,
  predictTimer: null,
  gapValues: null,
  gapScale: null,
  gapRows: [],
  gapAnimationFrame: null
};

function showError(message) {
  const alert = byId("alert");
  alert.textContent = message;
  alert.hidden = false;
}

function clearError() {
  byId("alert").hidden = true;
}

function validateVector(values) {
  if (!Array.isArray(values) || values.length !== 36 ||
      !values.every((value) => typeof value === "number" && Number.isFinite(value) &&
        value >= 0 && value <= 1)) {
    throw new Error("The design must be a JSON array of exactly 36 finite values in [0, 1].");
  }
  return values.slice();
}

function densityColor(value) {
  const low = [241, 247, 243];
  const high = [27, 71, 86];
  return "rgb(" + low.map((x, i) => Math.round(x + (high[i] - x) * value)).join(",") + ")";
}

function drawEditableOutline(ctx, cellSize, half) {
  ctx.beginPath();
  ctx.moveTo((2 * half - 1) * cellSize, 0);
  for (let row = 0; row < half; row++) {
    ctx.lineTo((2 * half - 1 - row) * cellSize, (row + 1) * cellSize);
    if (row + 1 < half) {
      ctx.lineTo((2 * half - 2 - row) * cellSize, (row + 1) * cellSize);
    }
  }
  ctx.lineTo(2 * half * cellSize, half * cellSize);
  ctx.lineTo(2 * half * cellSize, 0);
  ctx.closePath();
  ctx.strokeStyle = "#d7834b";
  ctx.lineWidth = 4;
  ctx.stroke();
}

function drawUnitCell() {
  if (!state.meta) return;
  const canvas = byId("cellCanvas");
  const ctx = canvas.getContext("2d");
  const grid = state.meta.full_index_grid;
  const size = canvas.width / grid.length;
  for (let row = 0; row < grid.length; row++) {
    for (let col = 0; col < grid.length; col++) {
      const editable = state.meta.editable_index_grid[row][col] !== null;
      ctx.fillStyle = densityColor(state.design[grid[row][col]]);
      ctx.fillRect(col * size, row * size, size, size);
      ctx.strokeStyle = editable ? "#d5a276" : "#d5e3dc";
      ctx.lineWidth = editable ? 1.2 : 0.6;
      ctx.strokeRect(col * size + 0.5, row * size + 0.5, size - 1, size - 1);
    }
  }
  ctx.strokeStyle = "#ffffffb0";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(canvas.width / 2, 0);
  ctx.lineTo(canvas.width / 2, canvas.height);
  ctx.moveTo(0, canvas.height / 2);
  ctx.lineTo(canvas.width, canvas.height / 2);
  ctx.stroke();
  drawEditableOutline(ctx, size, grid.length / 2);
  const [row, col] = state.selected;
  ctx.strokeStyle = "#ffca76";
  ctx.lineWidth = 4;
  ctx.strokeRect(col * size + 2, row * size + 2, size - 4, size - 4);
}

function selectedIndex() {
  if (!state.meta) return null;
  return state.meta.editable_index_grid[state.selected[0]][state.selected[1]];
}

function syncSelectedControls() {
  const index = selectedIndex();
  if (index === null) return;
  state.brush = state.design[index];
  byId("brushSlider").value = String(state.brush);
  byId("brushValue").textContent = state.brush.toFixed(2);
  byId("selectedCell").textContent = "Selected: row " + (state.selected[0] + 1) +
    ", column " + (state.selected[1] + 1) + " · design #" + (index + 1);
}

function updateHistoryButtons() {
  byId("undoButton").disabled = !state.undo.length;
  byId("redoButton").disabled = !state.redo.length;
}

function clearCurrentResults() {
  if (state.gapAnimationFrame !== null) cancelAnimationFrame(state.gapAnimationFrame);
  state.gapAnimationFrame = null;
  state.gapRows = [];
  state.prediction = null;
  state.fem = null;
  byId("femResults").hidden = true;
  byId("femLegend").hidden = true;
  byId("femButton").disabled = true;
  byId("gapSummary").textContent = "Updating band gaps…";
  byId("gapSummary").classList.toggle("has-open", false);
  byId("gapChart").replaceChildren();
  byId("gapScaleNegative").textContent = "−";
  byId("gapScalePositive").textContent = "+";
  byId("plotTooltip").hidden = true;
  byId("predictionTime").textContent = "—";
  byId("totalWait").textContent = "—";
  byId("predictionStatus").textContent = "Prediction pending…";
  byId("femStatus").textContent = state.femBusy
    ? "Verifying an earlier design; its result will be discarded."
    : "Run a prediction to enable verification.";
  renderPlot();
}

function updateDeviceLabel() {
  byId("devicePill").textContent = state.device.toUpperCase() + " · " + state.kPoints + " k POINTS";
}

function designChanged() {
  state.revision += 1;
  state.lastEditAt = performance.now();
  clearCurrentResults();
  drawUnitCell();
  syncSelectedControls();
  clearError();
  schedulePrediction(250);
}

function applyDesign(values, recordHistory = true) {
  const next = validateVector(values);
  if (state.design.length && next.every((value, index) => value === state.design[index])) return;
  if (recordHistory && state.design.length) {
    state.undo.push(state.design.slice());
    if (state.undo.length > 100) state.undo.shift();
    state.redo.length = 0;
  }
  state.design = next;
  updateHistoryButtons();
  designChanged();
}

function undo() {
  if (!state.undo.length) return;
  state.redo.push(state.design.slice());
  state.design = state.undo.pop();
  updateHistoryButtons();
  designChanged();
}

function redo() {
  if (!state.redo.length) return;
  state.undo.push(state.design.slice());
  state.design = state.redo.pop();
  updateHistoryButtons();
  designChanged();
}

function cellAt(event) {
  const canvas = byId("cellCanvas");
  const rect = canvas.getBoundingClientRect();
  const col = Math.floor((event.clientX - rect.left) / rect.width * 16);
  const row = Math.floor((event.clientY - rect.top) / rect.height * 16);
  if (row < 0 || col < 0 || row >= 16 || col >= 16) return null;
  return {row, col, index: state.meta.editable_index_grid[row][col]};
}

function paintCell(cell) {
  if (!cell || cell.index === null) return;
  state.selected = [cell.row, cell.col];
  if (state.design[cell.index] === state.brush) {
    drawUnitCell();
    syncSelectedControls();
    return;
  }
  state.design[cell.index] = state.brush;
  state.strokeChanged = true;
  designChanged();
}

function beginStroke(event) {
  if (!state.meta) return;
  const cell = cellAt(event);
  if (!cell || cell.index === null) return;
  event.preventDefault();
  state.drawing = true;
  state.strokeBase = state.design.slice();
  state.strokeChanged = false;
  state.lastPaintCell = cell;
  byId("cellCanvas").setPointerCapture(event.pointerId);
  paintCell(cell);
}

function moveStroke(event) {
  if (!state.drawing) return;
  const cell = cellAt(event);
  if (!cell) return;
  const previous = state.lastPaintCell;
  const steps = Math.max(Math.abs(cell.row - previous.row), Math.abs(cell.col - previous.col));
  for (let step = 1; step <= steps; step++) {
    const row = Math.round(previous.row + (cell.row - previous.row) * step / steps);
    const col = Math.round(previous.col + (cell.col - previous.col) * step / steps);
    paintCell({row, col, index: state.meta.editable_index_grid[row][col]});
  }
  state.lastPaintCell = cell;
}

function endStroke() {
  if (!state.drawing) return;
  state.drawing = false;
  if (state.strokeChanged) {
    state.undo.push(state.strokeBase);
    if (state.undo.length > 100) state.undo.shift();
    state.redo.length = 0;
    updateHistoryButtons();
    schedulePrediction(0);
  }
  state.strokeBase = null;
  state.lastPaintCell = null;
}

function schedulePrediction(delay) {
  state.predictPending = true;
  if (state.predictTimer !== null) clearTimeout(state.predictTimer);
  state.predictTimer = setTimeout(runPrediction, delay);
}

async function postJSON(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body)
  });
  const data = await response.json();
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : "Request failed (" + response.status + ").";
    throw new Error(detail);
  }
  return data;
}

async function runPrediction() {
  state.predictTimer = null;
  if (!state.predictPending || state.predictBusy || !state.meta) return;
  state.predictPending = false;
  state.predictBusy = true;
  const revision = state.revision;
  const design = state.design.slice();
  const kPoints = state.kPoints;
  const device = state.device;
  byId("predictionStatus").textContent = device === "cuda" && !state.cudaReady
    ? "Initializing CUDA model and solving bands…"
    : "Solving bands on " + device.toUpperCase() + "…";
  try {
    const result = await postJSON("/api/predict", {revision, design, k_points: kPoints, device});
    if (result.device === "cuda") state.cudaReady = true;
    if (result.revision === state.revision && result.k_points === state.kPoints &&
        result.device === state.device) {
      result.total_seconds = (performance.now() - state.lastEditAt) / 1000;
      state.prediction = result;
      showPrediction(result);
      clearError();
    }
  } catch (error) {
    if (revision === state.revision) {
      byId("predictionStatus").textContent = "Prediction failed.";
      showError(error.message);
    }
  } finally {
    state.predictBusy = false;
    if (state.predictPending) schedulePrediction(0);
  }
}

function formatFrequency(value) {
  return Number(value).toFixed(4);
}

function bandGaps(bands) {
  return Array.from({length: bands[0].length - 1}, (_, band) => {
    const lower = Math.max(...bands.map((row) => row[band]));
    const upper = Math.min(...bands.map((row) => row[band + 1]));
    return {band: band + 1, lower, upper, width: upper - lower};
  });
}

function formatGapWidth(width) {
  const value = Math.abs(width) < 0.0001 && width !== 0
    ? width.toExponential(1) : width.toFixed(4);
  return (width > 0 ? "+" : "") + value;
}

function gapMagnitude(values) {
  return Math.max(1e-6, ...values.map((value) => Math.abs(value)));
}

function paintGapChart(values, scale) {
  state.gapValues = values.slice();
  state.gapScale = scale;
  byId("gapScaleNegative").textContent = formatGapWidth(-scale);
  byId("gapScalePositive").textContent = formatGapWidth(scale);
  const openCount = values.filter((value) => value > 0).length;
  const summary = byId("gapSummary");
  summary.textContent = openCount
    ? openCount + " open gap" + (openCount === 1 ? "" : "s")
    : "No open gaps on this sampled path";
  summary.classList.toggle("has-open", openCount > 0);
  state.gapRows.forEach(({row, bar, value}, index) => {
    const width = values[index];
    const fraction = Math.min(0.5, Math.abs(width) / scale * 0.5);
    row.className = "gap-row " + (width > 0 ? "open" : width < 0 ? "negative" : "zero");
    row.setAttribute("aria-label", "Bands " + (index + 1) + " and " + (index + 2) +
      ": " + formatGapWidth(width) + (width > 0 ? " open gap" : " overlap or closed gap"));
    bar.style.left = ((width < 0 ? 0.5 - fraction : 0.5) * 100).toFixed(2) + "%";
    bar.style.width = (fraction * 100).toFixed(2) + "%";
    value.textContent = formatGapWidth(width);
  });
}

function renderGaps(bands) {
  const gaps = bandGaps(bands);
  const target = gaps.map((gap) => gap.width);
  const from = state.gapValues && state.gapValues.length === target.length
    ? state.gapValues.slice() : target.map(() => 0);
  const fromScale = state.gapScale || gapMagnitude(target);
  const targetScale = gapMagnitude(target);
  if (state.gapAnimationFrame !== null) cancelAnimationFrame(state.gapAnimationFrame);
  state.gapAnimationFrame = null;
  state.gapRows = gaps.map((gap) => {
    const row = document.createElement("div");
    row.setAttribute("role", "listitem");
    const pair = document.createElement("span");
    pair.className = "gap-pair";
    pair.textContent = gap.band + "–" + (gap.band + 1);
    const track = document.createElement("div");
    track.className = "gap-track";
    const bar = document.createElement("span");
    bar.className = "gap-bar";
    track.appendChild(bar);
    const value = document.createElement("strong");
    value.className = "gap-width";
    row.append(pair, track, value);
    return {row, bar, value};
  });
  byId("gapChart").replaceChildren(...state.gapRows.map(({row}) => row));
  paintGapChart(from, fromScale);
  if (target.every((width, index) => width === from[index]) && targetScale === fromScale) return gaps;
  const reducedMotion = typeof window !== "undefined" &&
    window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reducedMotion) {
    paintGapChart(target, targetScale);
    return gaps;
  }
  const started = performance.now();
  const animate = (now) => {
    const progress = Math.min(1, Math.max(0, (now - started) / GAP_ANIMATION_MS));
    const eased = 1 - (1 - progress) ** 3;
    const values = from.map((width, index) => width + (target[index] - width) * eased);
    const scale = fromScale + (targetScale - fromScale) * eased;
    paintGapChart(values, scale);
    state.gapAnimationFrame = progress < 1 ? requestAnimationFrame(animate) : null;
  };
  state.gapAnimationFrame = requestAnimationFrame(animate);
  return gaps;
}

function showPrediction(result) {
  renderGaps(result.bands);
  byId("predictionTime").textContent = result.prediction_seconds.toFixed(2) + " s";
  byId("totalWait").textContent = result.total_seconds === undefined
    ? "—" : result.total_seconds.toFixed(2) + " s";
  byId("predictionStatus").textContent = "Prediction ready · design version " + result.revision;
  byId("femButton").disabled = state.femBusy;
  byId("femStatus").textContent = state.femBusy
    ? "FEM is running for an earlier design."
    : "Ready to verify this design.";
  renderPlot();
}

function svgNode(tag, attributes, text) {
  const node = document.createElementNS(SVG_NS, tag);
  Object.entries(attributes || {}).forEach(([key, value]) => node.setAttribute(key, String(value)));
  if (text !== undefined) node.textContent = text;
  return node;
}

function closedRows(rows) {
  return rows.concat([rows[0]]);
}

function plotScale(predicted, verified) {
  const allValues = predicted.flat().concat(verified ? verified.flat() : []);
  const minValue = Math.min(...allValues);
  const maxValue = Math.max(...allValues);
  const range = Math.max(maxValue - minValue, 0.1);
  const yMin = Math.max(0, minValue - range * 0.05);
  const yMax = maxValue + range * 0.07;
  return {yMin, yMax, pointY: (value) =>
    PLOT.y1 - (value - yMin) / (yMax - yMin) * (PLOT.y1 - PLOT.y0)};
}

function nearestBand(result, verified, x, y) {
  const {x0, x1, y0, y1} = PLOT;
  if (x < x0 || x > x1 || y < y0 - 10 || y > y1 + 10) return null;
  const axisEnd = result.path_ticks[3];
  const position = (x - x0) / (x1 - x0) * axisEnd;
  const segment = Math.min(axisEnd - 1, Math.floor(position));
  const fraction = position - segment;
  const {pointY} = plotScale(result.bands, verified);
  let closest = null;
  for (const [source, rows] of [["DEN", result.bands], ["FEM", verified]]) {
    if (!rows) continue;
    const closed = closedRows(rows);
    for (let band = 0; band < rows[0].length; band++) {
      const frequency = closed[segment][band] * (1 - fraction) + closed[segment + 1][band] * fraction;
      const distance = Math.abs(y - pointY(frequency));
      if (!closest || distance < closest.distance) {
        closest = {source, band: band + 1, frequency, distance, atGamma: position === axisEnd};
      }
    }
  }
  return closest && closest.distance <= 10 ? closest : null;
}

function renderPlot() {
  const svg = byId("bandsPlot");
  svg.replaceChildren();
  const result = state.prediction;
  if (!result || result.revision !== state.revision) {
    svg.appendChild(svgNode("rect", {x: 44, y: 25, width: 814, height: 400, rx: 12, fill: "#f3f7f5"}));
    svg.appendChild(svgNode("text", {x: 450, y: 229, "text-anchor": "middle", fill: "#819a99", "font-size": 19},
      state.meta ? "Drawing a new band structure…" : "Loading example design…"));
    return;
  }
  const predicted = result.bands;
  const verified = state.fem && state.fem.revision === state.revision ? state.fem.fem_bands : null;
  const {yMin, yMax, pointY} = plotScale(predicted, verified);
  const {x0, x1, y0, y1} = PLOT;
  const axisEnd = result.path_ticks[3];
  const pointX = (i) => x0 + (x1 - x0) * i / axisEnd;

  svg.appendChild(svgNode("rect", {x: x0, y: y0, width: x1 - x0, height: y1 - y0, fill: "#fbfdfb"}));
  for (let i = 0; i <= 5; i++) {
    const value = yMin + (yMax - yMin) * i / 5;
    const y = pointY(value);
    svg.appendChild(svgNode("line", {x1: x0, x2: x1, y1: y, y2: y, stroke: "#e5eeea", "stroke-width": 1}));
    svg.appendChild(svgNode("text", {x: 55, y: y + 4, "text-anchor": "end", fill: "#71898d", "font-size": 12}, value.toFixed(2)));
  }
  const ticks = result.path_ticks;
  ["Γ", "X", "M", "Γ"].forEach((label, i) => {
    const x = pointX(ticks[i]);
    svg.appendChild(svgNode("line", {x1: x, x2: x, y1: y0, y2: y1, stroke: "#d7e5de", "stroke-width": 1}));
    svg.appendChild(svgNode("text", {x, y: 453, "text-anchor": "middle", fill: "#4a6871", "font-size": 16}, label));
  });
  bandGaps(predicted).filter((gap) => gap.width > 0).forEach((gap) => {
    const top = pointY(gap.upper);
    const bottom = pointY(gap.lower);
    svg.appendChild(svgNode("rect", {x: x0, y: top, width: x1 - x0, height: bottom - top,
      fill: "#d8eee5", opacity: .66}));
  });
  const drawSeries = (data, dashed) => {
    for (let band = 0; band < data[0].length; band++) {
      const points = closedRows(data).map((row, i) =>
        (i ? "L" : "M") + pointX(i).toFixed(1) + "," + pointY(row[band]).toFixed(1)).join(" ");
      svg.appendChild(svgNode("path", {
        d: points, fill: "none", stroke: dashed ? "#b77951" : BAND_COLORS[band % BAND_COLORS.length],
        "stroke-width": dashed ? 1.7 : 2.1,
        "stroke-opacity": dashed ? .82 : .95,
        "stroke-dasharray": dashed ? "5 5" : ""
      }));
    }
  };
  drawSeries(predicted, false);
  if (verified) drawSeries(verified, true);
  svg.appendChild(svgNode("rect", {x: x0, y: y0, width: x1 - x0, height: y1 - y0, fill: "none", stroke: "#bfd0cb", "stroke-width": 1}));
  svg.appendChild(svgNode("text", {x: 16, y: 240, fill: "#6c8184", "font-size": 12, transform: "rotate(-90 16 240)"},
    "Frequency  ωa / 2πc"));
}

function showFem(result) {
  state.fem = result;
  byId("femLegend").hidden = false;
  byId("femResults").hidden = false;
  byId("maeValue").textContent = result.metrics.mae.toFixed(5);
  byId("rmseValue").textContent = result.metrics.rmse.toFixed(5);
  const gapIou = result.metrics.mean_open_gap_iou;
  byId("iouValue").textContent = gapIou === null ? "n/a" : gapIou.toFixed(4);
  byId("femTime").textContent = result.fem_seconds.toFixed(1) + " s";
  byId("perBandErrors").replaceChildren(...result.metrics.per_band_mae.map((error, i) => {
    const item = document.createElement("div");
    item.className = "band-error";
    item.textContent = "Band " + (i + 1);
    const value = document.createElement("strong");
    value.textContent = error.toExponential(2);
    item.appendChild(value);
    return item;
  }));
  byId("femStatus").textContent = "FEM verified · design version " + result.revision;
  renderPlot();
}

async function runFem() {
  if (state.femBusy || !state.prediction || state.prediction.revision !== state.revision) return;
  const revision = state.revision;
  const design = state.design.slice();
  const kPoints = state.kPoints;
  const device = state.device;
  state.femBusy = true;
  byId("femButton").disabled = true;
  byId("femStatus").textContent = "Running FEM for design version " + revision + "…";
  try {
    const result = await postJSON("/api/fem", {revision, design, k_points: kPoints, device});
    if (result.revision === state.revision && result.k_points === state.kPoints &&
        result.device === state.device) {
      if (!state.prediction || state.prediction.revision !== state.revision) {
        state.prediction = {
          revision,
          device,
          k_points: result.k_points,
          path_ticks: result.path_ticks,
          bands: result.prediction_bands,
          prediction_seconds: result.prediction_seconds
        };
        showPrediction(state.prediction);
      }
      showFem(result);
      clearError();
    } else {
      byId("femStatus").textContent = "Earlier FEM result discarded. Verify the current design when ready.";
    }
  } catch (error) {
    if (revision === state.revision) {
      byId("femStatus").textContent = "FEM verification failed.";
      showError(error.message);
    }
  } finally {
    state.femBusy = false;
    byId("femButton").disabled = !state.prediction || state.prediction.revision !== state.revision;
  }
}

function editSelectedValue(value) {
  const next = Number(value);
  if (state.sliderBase === null) {
    state.sliderBase = state.design.slice();
    state.sliderChanged = false;
  }
  state.brush = next;
  byId("brushSlider").value = String(next);
  byId("brushValue").textContent = next.toFixed(2);
  const index = selectedIndex();
  if (index !== null && state.design[index] !== next) {
    state.design[index] = next;
    state.sliderChanged = true;
    designChanged();
  }
}

function finishSliderEdit() {
  if (state.sliderChanged) {
    state.undo.push(state.sliderBase);
    if (state.undo.length > 100) state.undo.shift();
    state.redo.length = 0;
    updateHistoryButtons();
    schedulePrediction(0);
  }
  state.sliderBase = null;
  state.sliderChanged = false;
}

function bindControls() {
  const cell = byId("cellCanvas");
  cell.addEventListener("pointerdown", beginStroke);
  cell.addEventListener("pointermove", moveStroke);
  cell.addEventListener("pointerup", endStroke);
  cell.addEventListener("pointercancel", endStroke);
  cell.addEventListener("keydown", (event) => {
    let [row, col] = state.selected;
    const moves = {ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1]};
    const delta = moves[event.key];
    if (!delta) return;
    event.preventDefault();
    row += delta[0];
    col += delta[1];
    while (row >= 0 && col >= 0 && row < 16 && col < 16) {
      if (state.meta.editable_index_grid[row][col] !== null) {
        state.selected = [row, col];
        drawUnitCell();
        syncSelectedControls();
        return;
      }
      row += delta[0];
      col += delta[1];
    }
  });
  byId("brushSlider").addEventListener("input", (event) => editSelectedValue(event.target.value));
  byId("brushSlider").addEventListener("change", finishSliderEdit);
  byId("brushSlider").addEventListener("blur", finishSliderEdit);
  byId("airButton").addEventListener("click", () => { editSelectedValue(0); finishSliderEdit(); });
  byId("materialButton").addEventListener("click", () => { editSelectedValue(1); finishSliderEdit(); });
  byId("kPointSelect").addEventListener("change", (event) => {
    state.kPoints = Number(event.target.value);
    updateDeviceLabel();
    byId("plotFootnote").textContent = "Γ → X → M → Γ · " + state.kPoints +
      " computed k points + closing Γ copied from start · 10 predicted bands · ωa / 2πc";
    designChanged();
    schedulePrediction(0);
  });
  byId("deviceSelect").addEventListener("change", (event) => {
    state.device = event.target.value;
    updateDeviceLabel();
    designChanged();
    schedulePrediction(0);
  });
  byId("undoButton").addEventListener("click", undo);
  byId("redoButton").addEventListener("click", redo);
  byId("exampleButton").addEventListener("click", () => applyDesign(state.example));
  byId("clearButton").addEventListener("click", () => applyDesign(Array(36).fill(0)));
  byId("randomButton").addEventListener("click", () => applyDesign(Array.from({length: 36}, () => Math.random() < .5 ? 0 : 1)));
  byId("femButton").addEventListener("click", runFem);
  byId("importButton").addEventListener("click", () => byId("importFile").click());
  byId("importFile").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      applyDesign(validateVector(JSON.parse(await file.text())));
    } catch (error) {
      showError("Import failed: " + error.message);
    }
    event.target.value = "";
  });
  byId("exportButton").addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(state.design, null, 2) + "\n"], {type: "application/json"});
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "tm_design.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  byId("bandsPlot").addEventListener("mousemove", (event) => {
    const result = state.prediction;
    if (!result || result.revision !== state.revision) return;
    const svg = event.currentTarget;
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const position = point.matrixTransform(svg.getScreenCTM().inverse());
    const verified = state.fem && state.fem.revision === state.revision
      ? state.fem.fem_bands : null;
    const nearest = nearestBand(result, verified, position.x, position.y);
    const tooltip = byId("plotTooltip");
    if (!nearest) {
      tooltip.hidden = true;
      return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    tooltip.textContent = "Band " + nearest.band + " · " + nearest.source +
      "\nωa / 2πc ≈ " + formatFrequency(nearest.frequency) +
      (nearest.atGamma ? "\nΓ (same as first k point)" : "");
    tooltip.style.left = Math.max(0, Math.min(event.clientX - rect.left + 12, rect.width - 190)) + "px";
    tooltip.style.top = Math.max(event.clientY - rect.top - 55, 5) + "px";
    tooltip.hidden = false;
  });
  byId("bandsPlot").addEventListener("mouseleave", () => { byId("plotTooltip").hidden = true; });
}

async function initialize() {
  try {
    const response = await fetch("/api/meta");
    if (!response.ok) throw new Error("Could not load solver metadata.");
    const meta = await response.json();
    if (meta.design_dim !== 36 || meta.grid_size !== 16 || meta.bands !== 10 ||
        meta.editable_index_grid.length !== 16 || meta.path_ticks.length !== 4 ||
        meta.k_point_options.length !== 8) {
      throw new Error("This interface expects the published 36-variable TM model.");
    }
    state.meta = meta;
    state.example = validateVector(meta.example_design);
    state.design = state.example.slice();
    state.kPoints = meta.k_points;
    state.device = meta.default_device || meta.device;
    state.cudaReady = Boolean(meta.cuda_ready);
    state.revision = 1;
    state.lastEditAt = performance.now();
    updateDeviceLabel();
    byId("deviceSelect").replaceChildren(...["cpu", "cuda"].map((device) => {
      const option = document.createElement("option");
      option.value = device;
      option.textContent = device.toUpperCase() + (device === "cuda" && !meta.available_devices.includes("cuda")
        ? " (unavailable)" : "");
      option.disabled = !meta.available_devices.includes(device);
      return option;
    }));
    byId("deviceSelect").value = state.device;
    byId("kPointSelect").replaceChildren(...meta.k_point_options.map((count) => {
      const option = document.createElement("option");
      option.value = String(count);
      option.textContent = String(count);
      return option;
    }));
    byId("kPointSelect").value = String(state.kPoints);
    bindControls();
    drawUnitCell();
    syncSelectedControls();
    renderPlot();
    schedulePrediction(0);
  } catch (error) {
    byId("predictionStatus").textContent = "Demo unavailable.";
    showError(error.message);
  }
}

initialize();
