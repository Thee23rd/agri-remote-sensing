const state = {
  crops: [],
  source: null,
  observations: [],
  cloudyPasses: null,
  result: null,
  mode: "viterbi",
  layout: null,
};

const $ = (id) => document.getElementById(id);

const dateLabel = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric" });

document.addEventListener("DOMContentLoaded", () => {
  $("season-form").addEventListener("submit", (event) => {
    event.preventDefault();
    simulate();
  });
  $("crop").addEventListener("change", () => {
    state.result = null;
    state.observations = [];
    state.source = null;
    state.cloudyPasses = null;
    $("learn").disabled = true;
    $("metrics").hidden = true;
    $("chart-empty").hidden = false;
    $("posterior-empty").hidden = false;
    $("ribbon").hidden = true;
    $("agreement").hidden = true;
    $("status").hidden = true;
    const empty = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 6;
    cell.className = "empty-cell";
    cell.textContent = "No passes yet.";
    empty.appendChild(cell);
    $("passes").replaceChildren(empty);
    syncCrop();
    draw();
  });
  $("csv").addEventListener("change", uploadCsv);
  $("mode-viterbi").addEventListener("click", () => setMode("viterbi"));
  $("mode-filtered").addEventListener("click", () => setMode("filtered"));
  $("learn").addEventListener("click", learn);
  $("series-chart").addEventListener("mousemove", hoverSeries);
  $("series-chart").addEventListener("mouseleave", () => {
    $("tooltip").hidden = true;
  });
  window.addEventListener("resize", draw);
  loadCrops();
});

async function loadCrops() {
  try {
    const payload = await getJson("/api/crops");
    state.crops = payload.crops;
    const select = $("crop");
    select.replaceChildren();
    for (const crop of state.crops) {
      const option = document.createElement("option");
      option.value = crop.id;
      option.textContent = crop.name;
      select.appendChild(option);
    }
    syncCrop();
  } catch (error) {
    showStatus(error.message, false);
  }
}

function selectedCrop() {
  return state.crops.find((crop) => crop.id === $("crop").value) || state.crops[0];
}

function syncCrop() {
  const crop = selectedCrop();
  if (!crop) return;
  $("planting").value = crop.default_planting;
  $("planting-note").textContent = crop.planting_note;
  const list = $("stage-list");
  list.replaceChildren();
  for (const stage of crop.stages) {
    const item = document.createElement("li");
    const dot = document.createElement("i");
    dot.className = "dot";
    dot.style.background = stage.color;
    const copy = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = stage.name;
    const summary = document.createElement("span");
    summary.textContent = stage.summary;
    copy.append(name, summary);
    item.append(dot, copy);
    list.appendChild(item);
  }
  if (!state.result) renderSignatures(crop.means, crop.stages, "Textbook prior");
}

async function simulate() {
  const seed = Math.floor(Math.random() * 1_000_000_000);
  const body = {
    crop: $("crop").value,
    planting_date: $("planting").value,
    interval_days: Number($("interval").value),
    cloud_prob: $("cloud").checked ? 0.18 : 0,
    seed,
  };
  await run(async () => {
    const season = await postJson("/api/simulate", body);
    state.source = "simulator";
    state.observations = season.observations;
    state.cloudyPasses = season.cloudy_passes;
    state.result = await postJson("/api/infer", {
      crop: body.crop,
      observations: season.observations,
    });
    showStatus(
      `Simulated ${season.clear_passes} clear passes` +
        (season.cloudy_passes ? ` and left out ${season.cloudy_passes} cloudy ones.` : "."),
      true
    );
  });
}

async function uploadCsv(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  try {
    const text = await file.text();
    const observations = parseCsv(text);
    await run(async () => {
      state.source = "upload";
      state.observations = observations;
      state.cloudyPasses = null;
      state.result = await postJson("/api/infer", {
        crop: $("crop").value,
        observations,
      });
      showStatus(`Scored ${observations.length} passes from ${file.name}.`, true);
    });
  } catch (error) {
    showStatus(error.message, false);
  }
}

async function learn() {
  if (!state.observations.length) return;
  await run(async () => {
    const trained = await postJson("/api/calibrate", {
      crop: $("crop").value,
      observations: state.observations,
      source: state.source || "upload",
      n_fields: 28,
      seed: 7,
    });
    state.result = trained.adapted_inference;
    const prior = trained.agreement_prior;
    const adapted = trained.agreement_adapted;
    let note = trained.note;
    if (prior != null && adapted != null) {
      note += ` Simulator agreement ${percent(prior)} with the textbook signature, ${percent(adapted)} after learning.`;
    }
    showStatus(note, true);
  });
}

async function run(work) {
  setBusy(true);
  try {
    await work();
    $("learn").disabled = state.observations.length === 0;
    render();
  } catch (error) {
    showStatus(error.message, false);
  } finally {
    setBusy(false);
  }
}

function setMode(mode) {
  state.mode = mode;
  $("mode-viterbi").classList.toggle("is-active", mode === "viterbi");
  $("mode-filtered").classList.toggle("is-active", mode === "filtered");
  $("mode-viterbi").setAttribute("aria-pressed", String(mode === "viterbi"));
  $("mode-filtered").setAttribute("aria-pressed", String(mode === "filtered"));
  $("mode-copy").textContent = mode === "viterbi"
    ? "The single most likely stage sequence given every clear pass."
    : "The stage belief on each date using only the passes that had arrived by then.";
  render();
}

function render() {
  const result = state.result;
  const hasResult = Boolean(result);
  $("metrics").hidden = !hasResult;
  $("chart-empty").hidden = hasResult;
  $("posterior-empty").hidden = hasResult;
  $("ribbon").hidden = !hasResult;
  if (!hasResult) {
    draw();
    return;
  }

  const latest = result.latest;
  const stageKey = state.mode === "viterbi" ? "stage_viterbi" : "stage_filtered";
  const confidenceKey = state.mode === "viterbi" ? "confidence_viterbi" : "confidence_filtered";
  const stage = result.stages.find((item) => item.id === latest[stageKey]);
  $("stage-label").textContent = state.mode === "viterbi" ? "Latest reconstructed stage" : "Stage known on the latest pass";
  $("metric-stage").textContent = stage ? stage.name : "—";
  $("metric-confidence").textContent = `${percent(latest[confidenceKey])} belief`;
  $("metric-passes").textContent = String(result.clear_passes);
  $("metric-cloud").textContent = state.cloudyPasses
    ? `${state.cloudyPasses} cloudy passes omitted`
    : result.warning || "Indices only, no field visit";
  $("metric-peak").textContent = result.peak.value.toFixed(2);
  $("metric-peak-date").textContent = dateLabel.format(parseDate(result.peak.date));
  $("metric-senescence").textContent = result.senescence_onset
    ? dateLabel.format(parseDate(result.senescence_onset))
    : "Not reached";

  $("belief-title").textContent = state.mode === "viterbi" ? "Stage belief, full season" : "Stage belief, as of each date";
  $("signature-label").textContent = result.model_label;
  renderSignatures(result.means, result.stages, result.model_label);
  renderRibbon(result);
  renderPasses(result);
  const agreement = $("agreement");
  if (result.agreement == null) {
    agreement.hidden = true;
  } else {
    agreement.hidden = false;
    agreement.textContent = `Agreement with the stages the simulator planted: ${percent(result.agreement)}. Real fields do not come with this check.`;
  }
  draw();
}

function renderSignatures(means, stages, label) {
  $("signature-label").textContent = label;
  const body = $("signatures");
  body.replaceChildren();
  stages.forEach((stage, index) => {
    const row = document.createElement("tr");
    const name = document.createElement("th");
    name.scope = "row";
    name.textContent = stage.name;
    row.appendChild(name);
    for (const value of means[index]) {
      const cell = document.createElement("td");
      cell.textContent = Number(value).toFixed(2);
      row.appendChild(cell);
    }
    body.appendChild(row);
  });
}

function renderRibbon(result) {
  const key = state.mode === "viterbi" ? "stage_viterbi" : "stage_filtered";
  const runs = [];
  for (const row of result.timeline) {
    const last = runs[runs.length - 1];
    if (last && last.id === row[key]) last.count += 1;
    else runs.push({ id: row[key], count: 1 });
  }
  const ribbon = $("ribbon");
  ribbon.replaceChildren();
  for (const run of runs) {
    const stage = result.stages.find((item) => item.id === run.id);
    const cell = document.createElement("span");
    cell.style.flex = String(run.count);
    cell.style.background = stage.color;
    cell.style.color = luminance(stage.color) > 160 ? "#17323c" : "white";
    cell.title = stage.name;
    cell.textContent = run.count >= 2 ? stage.short : "";
    ribbon.appendChild(cell);
  }
}

function renderPasses(result) {
  const stageKey = state.mode === "viterbi" ? "stage_viterbi" : "stage_filtered";
  const confidenceKey = state.mode === "viterbi" ? "confidence_viterbi" : "confidence_filtered";
  const names = Object.fromEntries(result.stages.map((stage) => [stage.id, stage.name]));
  const body = $("passes");
  body.replaceChildren();
  for (const row of result.timeline) {
    const tr = document.createElement("tr");
    for (const value of [
      dateLabel.format(parseDate(row.date)),
      formatIndex(row.ndvi),
      formatIndex(row.evi),
      formatIndex(row.ndre),
      names[row[stageKey]],
      percent(row[confidenceKey]),
    ]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      tr.appendChild(cell);
    }
    body.appendChild(tr);
  }
}

function draw() {
  drawSeries();
  drawPosterior();
}

function drawSeries() {
  const canvas = $("series-chart");
  const result = state.result;
  const box = prepareCanvas(canvas);
  if (!box || !result) {
    state.layout = null;
    return;
  }
  const { ctx, width, height } = box;
  const pad = { l: 44, r: 12, t: 16, b: 32 };
  const rows = result.timeline;
  const times = rows.map((row) => parseDate(row.date).getTime());
  const span = Math.max(times[times.length - 1] - times[0], 1);
  const plotW = width - pad.l - pad.r;
  const plotH = height - pad.t - pad.b;
  const xOf = (time) => pad.l + ((time - times[0]) / span) * plotW;
  const yOf = (value) => pad.t + (1 - (value + 0.08) / 1.08) * plotH;

  const stageKey = state.mode === "viterbi" ? "stage_viterbi" : "stage_filtered";
  for (let index = 0; index < rows.length; index += 1) {
    const stage = result.stages.find((item) => item.id === rows[index][stageKey]);
    const left = index === 0 ? pad.l : (xOf(times[index - 1]) + xOf(times[index])) / 2;
    const right = index === rows.length - 1 ? width - pad.r : (xOf(times[index]) + xOf(times[index + 1])) / 2;
    ctx.fillStyle = hexAlpha(stage.color, 0.16);
    ctx.fillRect(left, pad.t, right - left, plotH);
  }

  ctx.strokeStyle = "#e1ebe6";
  ctx.fillStyle = "#5c7278";
  ctx.lineWidth = 1;
  ctx.font = "11px Avenir Next, Segoe UI, sans-serif";
  for (const tick of [0, 0.25, 0.5, 0.75, 1]) {
    const y = yOf(tick);
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(width - pad.r, y);
    ctx.stroke();
    ctx.fillText(tick.toFixed(2), 6, y + 3);
  }

  const series = [
    ["ndvi", "#17323c", 2.4],
    ["evi", "#1d7a68", 1.7],
    ["ndre", "#c47b16", 1.7],
  ];
  for (const [key, color, widthPx] of series) {
    ctx.beginPath();
    let drawing = false;
    rows.forEach((row, index) => {
      if (row[key] == null) {
        drawing = false;
        return;
      }
      const x = xOf(times[index]);
      const y = yOf(row[key]);
      if (!drawing) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
      drawing = true;
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = widthPx;
    ctx.stroke();
  }

  drawDateLabels(ctx, rows, (index) => xOf(times[index]), height - 10);

  state.layout = { pad, times, xOf, rows, width };
}

function drawPosterior() {
  const canvas = $("posterior-chart");
  const result = state.result;
  const box = prepareCanvas(canvas);
  if (!box || !result) return;
  const { ctx, width, height } = box;
  const pad = { l: 64, r: 10, t: 10, b: 28 };
  const rows = result.timeline;
  const stages = result.stages;
  const plotW = width - pad.l - pad.r;
  const plotH = height - pad.t - pad.b;
  const cellW = plotW / rows.length;
  const cellH = plotH / stages.length;
  const key = state.mode === "viterbi" ? "posterior_smoothed" : "posterior_filtered";

  rows.forEach((row, timeIndex) => {
    stages.forEach((stage, stageIndex) => {
      const probability = row[key][stageIndex];
      ctx.fillStyle = hexAlpha(stage.color, 0.08 + 0.92 * probability);
      ctx.fillRect(
        pad.l + timeIndex * cellW,
        pad.t + (stages.length - 1 - stageIndex) * cellH,
        cellW + 0.6,
        cellH + 0.6
      );
    });
  });

  ctx.fillStyle = "#5c7278";
  ctx.font = "11px Avenir Next, Segoe UI, sans-serif";
  stages.forEach((stage, stageIndex) => {
    const y = pad.t + (stages.length - 1 - stageIndex) * cellH + cellH / 2 + 4;
    ctx.fillText(stage.short, 8, y);
  });
  drawDateLabels(ctx, rows, (index) => pad.l + (index + 0.5) * cellW, height - 8);
}

function hoverSeries(event) {
  if (!state.layout || !state.result) return;
  const rect = event.currentTarget.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const { pad, times, rows, width } = state.layout;
  if (x < pad.l || x > width - pad.r) {
    $("tooltip").hidden = true;
    return;
  }
  let nearest = 0;
  let best = Infinity;
  times.forEach((time, index) => {
    const distance = Math.abs(state.layout.xOf(time) - x);
    if (distance < best) {
      best = distance;
      nearest = index;
    }
  });
  const row = rows[nearest];
  const stageKey = state.mode === "viterbi" ? "stage_viterbi" : "stage_filtered";
  const confidenceKey = state.mode === "viterbi" ? "confidence_viterbi" : "confidence_filtered";
  const stage = state.result.stages.find((item) => item.id === row[stageKey]);
  const tip = $("tooltip");
  tip.hidden = false;
  tip.replaceChildren();
  const title = document.createElement("strong");
  title.textContent = `${dateLabel.format(parseDate(row.date))} · ${stage.name}`;
  tip.appendChild(title);
  tip.append(
    textLine(`NDVI ${formatIndex(row.ndvi)}   EVI ${formatIndex(row.evi)}   NDRE ${formatIndex(row.ndre)}`),
    textLine(`${percent(row[confidenceKey])} belief`)
  );
  const left = Math.min(x, rect.width - 200);
  tip.style.left = `${left}px`;
  tip.style.top = "12px";
}

function drawDateLabels(ctx, rows, xAt, y) {
  const labelCount = Math.min(5, rows.length);
  ctx.fillStyle = "#5c7278";
  ctx.font = "11px Avenir Next, Segoe UI, sans-serif";
  for (let tick = 0; tick < labelCount; tick += 1) {
    const index = Math.round((tick * (rows.length - 1)) / Math.max(labelCount - 1, 1));
    ctx.textAlign = tick === 0 ? "left" : tick === labelCount - 1 ? "right" : "center";
    ctx.fillText(shortDate(rows[index].date), xAt(index), y);
  }
  ctx.textAlign = "left";
}

function prepareCanvas(canvas) {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (width < 20 || height < 20) return null;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/).filter((line) => line.trim());
  if (lines.length < 2) throw new Error("The CSV needs a header and at least one pass.");
  const header = lines[0].split(",").map((cell) => cell.trim().toLowerCase());
  const dateIndex = header.indexOf("date");
  const ndviIndex = header.indexOf("ndvi");
  if (dateIndex < 0 || ndviIndex < 0) {
    throw new Error("The CSV needs date and ndvi columns. evi and ndre are optional.");
  }
  const eviIndex = header.indexOf("evi");
  const ndreIndex = header.indexOf("ndre");
  return lines.slice(1).map((line) => {
    const cells = line.split(",").map((cell) => cell.trim());
    return {
      date: cells[dateIndex],
      ndvi: numberOrNull(cells[ndviIndex]),
      evi: eviIndex >= 0 ? numberOrNull(cells[eviIndex]) : null,
      ndre: ndreIndex >= 0 ? numberOrNull(cells[ndreIndex]) : null,
    };
  });
}

function numberOrNull(value) {
  if (value == null || value === "") return null;
  const number = Number(value);
  if (Number.isNaN(number)) throw new Error(`Could not read '${value}' as a spectral index.`);
  return number;
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

async function errorMessage(response) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((item) => item.msg).join(" ");
  } catch (_error) {
    return "The request failed.";
  }
  return "The request failed.";
}

function showStatus(message, ok) {
  const status = $("status");
  status.hidden = false;
  status.textContent = message;
  status.classList.toggle("is-info", ok);
}

function setBusy(busy) {
  $("simulate").disabled = busy;
  $("learn").disabled = busy || state.observations.length === 0;
  document.body.setAttribute("aria-busy", String(busy));
}

function parseDate(value) {
  return new Date(`${value}T00:00:00`);
}

function shortDate(value) {
  return new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short" }).format(parseDate(value));
}

function formatIndex(value) {
  return value == null ? "—" : Number(value).toFixed(2);
}

function percent(value) {
  return `${Math.round(Number(value) * 100)}%`;
}

function luminance(hex) {
  const red = parseInt(hex.slice(1, 3), 16);
  const green = parseInt(hex.slice(3, 5), 16);
  const blue = parseInt(hex.slice(5, 7), 16);
  return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
}

function hexAlpha(hex, alpha) {
  const red = parseInt(hex.slice(1, 3), 16);
  const green = parseInt(hex.slice(3, 5), 16);
  const blue = parseInt(hex.slice(5, 7), 16);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function textLine(value) {
  const line = document.createElement("div");
  line.textContent = value;
  return line;
}
