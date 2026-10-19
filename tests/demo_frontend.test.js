"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../demo/static/app.js"), "utf8")
  .replace(/\binitialize\(\);\s*$/, "");
const elements = new Map();
const timers = new Map();
const animationFrames = new Map();
const requests = [];
let nextTimer = 1;
let nextFrame = 1;
let exportedBlob;

function node(tag = "div") {
  return {
      tag,
      hidden: false,
      disabled: false,
      textContent: "",
      value: "",
      style: {},
      attributes: {},
      children: [],
      listeners: {},
      classList: {toggle() {}},
      addEventListener(name, handler) { this.listeners[name] = handler; },
      setAttribute(name, value) { this.attributes[name] = value; },
      appendChild(child) { this.children.push(child); },
      append(...children) { this.children.push(...children); },
      replaceChildren(...children) { this.children = children; },
      click() {},
    };
}

function element(id) {
  if (!elements.has(id)) elements.set(id, node());
  return elements.get(id);
}

const context = vm.createContext({
  Blob,
  performance: {now: () => 1000},
  document: {
    getElementById: element,
    createElement: (tag) => node(tag),
    createElementNS: (_, tag) => node(tag),
  },
  URL: {
    createObjectURL(blob) { exportedBlob = blob; return "blob:design"; },
    revokeObjectURL() {},
  },
  setTimeout(fn) { const id = nextTimer++; timers.set(id, fn); return id; },
  clearTimeout(id) { timers.delete(id); },
  requestAnimationFrame(fn) { const id = nextFrame++; animationFrames.set(id, fn); return id; },
  cancelAnimationFrame(id) { animationFrames.delete(id); },
  fetch(url, options) {
    return new Promise((resolve) => requests.push({url, body: JSON.parse(options.body), resolve}));
  },
});
vm.runInContext(source, context);
vm.runInContext(`
  state.meta = {device: "cpu", editable_index_grid: Array.from({length: 16},
    () => Array(16).fill(null)), full_index_grid: [], path_ticks: [0, 10, 20, 30]};
  state.meta.editable_index_grid[0][15] = 0;
  state.design = Array(36).fill(0);
  state.example = Array(36).fill(0);
  state.revision = 1;
  globalThis.originalRenderPlot = renderPlot;
  renderPlot = () => {};
  drawWedge = () => {};
  drawUnitCell = () => {};
`, context);

function state() { return vm.runInContext("state", context); }
function flushAnimation(time) {
  const pending = [...animationFrames.values()];
  animationFrames.clear();
  pending.forEach((frame) => frame(time));
}
function respond(request, body) {
  request.resolve({ok: true, json: async () => body});
}
function result(revision, kPoints = 30, device = "cpu") {
  return {
    revision,
    device,
    k_points: kPoints,
    path_ticks: [0, kPoints / 3, 2 * kPoints / 3, kPoints],
    bands: Array.from({length: kPoints}, () => Array(10).fill(1)),
    prediction_seconds: 0.4,
  };
}

async function main() {
  const outline = [];
  context.outlineCtx = {
    beginPath() { outline.push(["begin"]); },
    moveTo(x, y) { outline.push(["move", x, y]); },
    lineTo(x, y) { outline.push(["line", x, y]); },
    closePath() { outline.push(["close"]); },
    stroke() { outline.push(["stroke"]); },
  };
  vm.runInContext("drawEditableOutline(outlineCtx, 32, 8)", context);
  assert.deepEqual(outline.slice(0, 5), [
    ["begin"], ["move", 480, 0], ["line", 480, 32],
    ["line", 448, 32], ["line", 448, 64],
  ]);
  assert.deepEqual(outline.slice(-4), [
    ["line", 512, 256], ["line", 512, 0], ["close"], ["stroke"],
  ]);

  const sampleBands = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0.1, 2.2, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1],
  ];
  context.sampleBands = sampleBands;
  const gaps = vm.runInContext("bandGaps(sampleBands)", context);
  assert.equal(gaps.length, 9);
  assert.ok(Math.abs(gaps[0].width - 0.9) < 1e-10);
  assert.ok(Math.abs(gaps[1].width + 0.2) < 1e-10);
  vm.runInContext("renderGaps(sampleBands)", context);
  assert.equal(elements.get("gapChart").children.length, 9);
  assert.ok(state().gapValues.every((value) => value === 0));
  flushAnimation(1650);
  assert.equal(elements.get("gapChart").children[0].className, "gap-row open");
  assert.equal(elements.get("gapChart").children[1].className, "gap-row negative");
  assert.equal(state().gapRows[0].bar.style.left, "50.00%");
  assert.ok(parseFloat(state().gapRows[1].bar.style.left) < 50);
  assert.ok(state().gapRows.every(({bar}) => parseFloat(bar.style.width) <= 50));
  assert.match(elements.get("gapSummary").textContent, /8 open gaps/);
  context.largerBands = sampleBands.map((row) => row.map((value, band) =>
    band === 1 ? value + 1.5 : value));
  const previous = Array.from(state().gapValues);
  vm.runInContext("renderGaps(largerBands)", context);
  assert.deepEqual(Array.from(state().gapValues), previous, "new bars start at cached old values");
  flushAnimation(1325);
  const halfway = state().gapValues.slice();
  assert.ok(halfway[0] > previous[0]);
  assert.ok(halfway[0] < vm.runInContext("bandGaps(largerBands)[0].width", context));
  assert.ok(state().gapRows.every(({bar}) => parseFloat(bar.style.width) <= 50),
    "interpolated bars must stay inside the symmetric scale");
  flushAnimation(1650);
  assert.ok(Math.abs(state().gapValues[0] - 2.4) < 1e-10);
  assert.equal(state().gapScale, 2.4);
  context.flippedBands = sampleBands.map((row) => row.map((value, band) =>
    band === 1 ? value - 1.5 : value));
  vm.runInContext("renderGaps(flippedBands)", context);
  assert.equal(state().gapRows[0].row.className, "gap-row open");
  flushAnimation(1325);
  assert.ok(state().gapValues[0] < 0, "the numeric value must cross zero smoothly");
  assert.equal(state().gapRows[0].row.className, "gap-row negative");
  assert.ok(parseFloat(state().gapRows[0].bar.style.left) < 50);
  flushAnimation(1650);
  vm.runInContext("renderGaps(sampleBands)", context);
  flushAnimation(1325);
  const interrupted = Array.from(state().gapValues);
  vm.runInContext("clearCurrentResults()", context);
  assert.equal(animationFrames.size, 0);
  assert.equal(elements.get("gapChart").children.length, 0);
  assert.deepEqual(Array.from(state().gapValues), interrupted,
    "an edited design keeps only the in-memory gap snapshot");
  vm.runInContext("renderGaps(largerBands)", context);
  assert.deepEqual(Array.from(state().gapValues), interrupted,
    "the next animation starts from the interrupted visual values");
  flushAnimation(1650);

  const hoverResult = result(1, 2);
  hoverResult.bands = sampleBands;
  context.hoverResult = hoverResult;
  assert.equal(vm.runInContext("closedRows(sampleBands).length", context), 3);
  assert.equal(vm.runInContext("closedRows(sampleBands)[2] === sampleBands[0]", context), true);
  const hover = vm.runInContext(`nearestBand(hoverResult, null, PLOT.x0,
    plotScale(sampleBands, null).pointY(sampleBands[0][3]))`, context);
  assert.equal(hover.band, 4);
  const gamma = vm.runInContext(`nearestBand(hoverResult, null, PLOT.x1,
    plotScale(sampleBands, null).pointY(sampleBands[0][3]))`, context);
  assert.equal(gamma.band, 4);
  assert.equal(gamma.frequency, sampleBands[0][3]);
  assert.equal(gamma.atGamma, true);
  context.femBands = sampleBands.map((row) => row.map((frequency) => frequency + 0.1));
  const femHover = vm.runInContext(`nearestBand(hoverResult, femBands, PLOT.x0,
    plotScale(sampleBands, femBands).pointY(femBands[0][3]))`, context);
  assert.equal(femHover.source, "FEM");
  assert.equal(femHover.band, 4);
  assert.equal(vm.runInContext("nearestBand(hoverResult, null, PLOT.x0, PLOT.y0 - 30)", context), null);
  vm.runInContext("state.prediction = hoverResult; originalRenderPlot()", context);
  const firstPath = elements.get("bandsPlot").children.find((child) => child.tag === "path");
  assert.ok(firstPath);
  const vertices = firstPath.attributes.d.split(" ");
  assert.equal(vertices.length, 3, "the plotted path must contain the closing Γ point");
  assert.equal(vertices[0].slice(vertices[0].indexOf(",")),
    vertices[2].slice(vertices[2].indexOf(",")), "the closing Γ must reuse the first frequency");
  state().prediction = null;

  assert.equal(vm.runInContext("validateVector(Array(36).fill(0.5)).length", context), 36);
  for (const bad of [Array(35).fill(0), [NaN, ...Array(35).fill(0)],
    [true, ...Array(35).fill(0)], [2, ...Array(35).fill(0)]]) {
    assert.throws(() => vm.runInContext("validateVector(input)", vm.createContext({
      input: bad,
      validateVector: vm.runInContext("validateVector", context),
    })));
  }

  state().predictPending = true;
  const first = vm.runInContext("runPrediction()", context);
  assert.equal(requests.length, 1);
  vm.runInContext("applyDesign(Array(36).fill(0.25)); applyDesign(Array(36).fill(0.75));", context);
  assert.equal(state().predictPending, true);
  assert.equal(requests.length, 1, "edits must coalesce while a prediction is in flight");
  respond(requests[0], result(1));
  await first;
  assert.equal(state().prediction, null, "a stale prediction must not be displayed");
  assert.equal(timers.size, 1);
  const next = [...timers.values()][0]();
  assert.equal(requests.length, 2);
  assert.equal(requests[1].body.revision, 3);
  assert.equal(requests[1].body.k_points, 30);
  assert.equal(requests[1].body.device, "cpu");
  assert.equal(requests[1].body.design[0], 0.75);
  respond(requests[1], result(3));
  await next;
  assert.equal(state().prediction.revision, 3);

  const fem = vm.runInContext("runFem()", context);
  assert.equal(requests[2].url, "/api/fem");
  vm.runInContext("applyDesign(Array(36).fill(0.4));", context);
  respond(requests[2], {...result(3), fem_bands: result(3).bands, metrics: {}});
  await fem;
  assert.equal(state().fem, null, "a stale FEM result must not be displayed");

  vm.runInContext("bindControls()", context);
  elements.get("exportButton").listeners.click();
  assert.deepEqual(JSON.parse(await exportedBlob.text()), Array(36).fill(0.4));
  await elements.get("importFile").listeners.change({
    target: {files: [{text: async () => JSON.stringify(Array(36).fill(0.6))}], value: "test"},
  });
  assert.deepEqual(Array.from(state().design), Array(36).fill(0.6));
  const previousUndo = state().undo.length;
  elements.get("brushSlider").listeners.input({target: {value: "0.2"}});
  elements.get("brushSlider").listeners.input({target: {value: "0.4"}});
  elements.get("brushSlider").listeners.change();
  assert.equal(state().design[0], 0.4);
  assert.equal(state().undo.length, previousUndo + 1, "one slider drag is one undo step");
  elements.get("kPointSelect").listeners.change({target: {value: "9"}});
  assert.equal(state().kPoints, 9);
  const sampled = vm.runInContext("runPrediction()", context);
  assert.equal(requests.at(-1).body.k_points, 9);
  respond(requests.at(-1), result(state().revision, 9));
  await sampled;
  assert.equal(state().prediction.bands.length, 9);
  elements.get("deviceSelect").listeners.change({target: {value: "cuda"}});
  assert.equal(state().device, "cuda");
  assert.equal(state().prediction, null);
  const cudaPrediction = vm.runInContext("runPrediction()", context);
  assert.equal(requests.at(-1).body.device, "cuda");
  respond(requests.at(-1), result(state().revision, 9, "cuda"));
  await cudaPrediction;
  assert.equal(state().prediction.device, "cuda");
  assert.equal(state().cudaReady, true);
  console.log("Frontend logic passed: staircase outline, bidirectional animated gaps, Γ closure, hover, coalescing, controls, JSON.");
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
