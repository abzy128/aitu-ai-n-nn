const modelSelect = document.querySelector("#modelSelect");
const startDateInput = document.querySelector("#startDate");
const endDateInput = document.querySelector("#endDate");
const refreshButton = document.querySelector("#refreshButton");
const statusLine = document.querySelector("#statusLine");
const datasetRange = document.querySelector("#datasetRange");
const canvas = document.querySelector("#chart");
const tooltip = document.querySelector("#tooltip");
const summaryModel = document.querySelector("#summaryModel");
const summaryPoints = document.querySelector("#summaryPoints");
const summaryAnomalies = document.querySelector("#summaryAnomalies");
const summaryThreshold = document.querySelector("#summaryThreshold");

const state = {
  points: [],
  plotPoints: [],
};

function toLocalInputValue(value) {
  const date = new Date(value);
  const pad = (num) => String(num).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return Number(value).toFixed(digits);
}

async function fetchJson(url) {
  const response = await fetch(url);
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.detail || `Request failed: ${response.status}`);
  }
  return body;
}

async function init() {
  try {
    statusLine.textContent = "Loading models...";
    const health = await fetchJson("/health");
    const models = await fetchJson("/api/models");
    datasetRange.textContent = `${health.datasetMinDateTime} to ${health.datasetMaxDateTime}`;

    modelSelect.innerHTML = "";
    for (const model of models.models) {
      const option = document.createElement("option");
      option.value = model.name;
      option.textContent = model.is_default ? `${model.name} (best F1)` : model.name;
      modelSelect.append(option);
    }
    modelSelect.value = models.defaultModel;

    const minDate = new Date(health.datasetMinDateTime);
    const initialEnd = new Date(minDate.getTime() + 24 * 60 * 60 * 1000);
    startDateInput.value = toLocalInputValue(minDate);
    endDateInput.value = toLocalInputValue(initialEnd);
    await loadPredictions();
  } catch (error) {
    statusLine.textContent = error.message;
  }
}

async function loadPredictions() {
  try {
    statusLine.textContent = "Loading predictions...";
    const params = new URLSearchParams({
      startDate: startDateInput.value,
      endDate: endDateInput.value,
      modelName: modelSelect.value,
    });
    const result = await fetchJson(`/api/predictions?${params.toString()}`);
    state.points = result.points;
    summaryModel.textContent = result.modelName;
    summaryPoints.textContent = result.summary.totalPoints;
    summaryAnomalies.textContent = result.summary.anomalyCount;
    summaryThreshold.textContent = formatNumber(result.threshold, 6);
    statusLine.textContent = result.points.length ? "Ready." : "No scored points in this range.";
    drawChart();
  } catch (error) {
    statusLine.textContent = error.message;
  }
}

function drawChart() {
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const padding = { left: 58, right: 24, top: 22, bottom: 42 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  ctx.strokeStyle = "#e3e8ef";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 5; i += 1) {
    const y = padding.top + (plotHeight * i) / 5;
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
  }
  ctx.stroke();

  if (!state.points.length) {
    state.plotPoints = [];
    ctx.fillStyle = "#667085";
    ctx.textAlign = "center";
    ctx.fillText("No data for selected range", width / 2, height / 2);
    return;
  }

  const values = state.points.map((point) => point.activePower);
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const valueRange = Math.max(0.001, maxValue - minValue);
  const xForIndex = (index) =>
    padding.left + (plotWidth * index) / Math.max(1, state.points.length - 1);
  const yForValue = (value) =>
    padding.top + plotHeight - ((value - minValue) / valueRange) * plotHeight;

  ctx.strokeStyle = "#2563eb";
  ctx.lineWidth = 2;
  ctx.beginPath();
  state.plotPoints = state.points.map((point, index) => ({
    ...point,
    x: xForIndex(index),
    y: yForValue(point.activePower),
  }));
  state.plotPoints.forEach((point, index) => {
    if (index === 0) ctx.moveTo(point.x, point.y);
    else ctx.lineTo(point.x, point.y);
  });
  ctx.stroke();

  for (const point of state.plotPoints) {
    if (!point.isAnomaly) continue;
    ctx.fillStyle = "#dc2626";
    ctx.beginPath();
    ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = "#667085";
  ctx.font = "12px sans-serif";
  ctx.textAlign = "right";
  ctx.fillText(formatNumber(maxValue, 2), padding.left - 8, padding.top + 4);
  ctx.fillText(formatNumber(minValue, 2), padding.left - 8, padding.top + plotHeight);
  ctx.textAlign = "left";
  ctx.fillText(state.points[0].dateTime.replace("T", " "), padding.left, height - 14);
  ctx.textAlign = "right";
  ctx.fillText(
    state.points[state.points.length - 1].dateTime.replace("T", " "),
    width - padding.right,
    height - 14,
  );
}

function showTooltip(event) {
  if (!state.plotPoints.length) return;
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  let nearest = state.plotPoints[0];
  let nearestDistance = Math.abs(nearest.x - x);
  for (const point of state.plotPoints) {
    const distance = Math.abs(point.x - x);
    if (distance < nearestDistance) {
      nearest = point;
      nearestDistance = distance;
    }
  }
  if (nearestDistance > 18) {
    tooltip.hidden = true;
    return;
  }
  tooltip.innerHTML = `
    <strong>${nearest.dateTime.replace("T", " ")}</strong><br>
    active_power: ${formatNumber(nearest.activePower, 3)}<br>
    score: ${formatNumber(nearest.score, 6)}<br>
    ${nearest.isAnomaly ? "Anomaly" : "Normal"}
  `;
  tooltip.style.left = `${Math.min(rect.width - 270, Math.max(8, nearest.x + 14))}px`;
  tooltip.style.top = `${Math.max(8, nearest.y - 18)}px`;
  tooltip.hidden = false;
}

refreshButton.addEventListener("click", loadPredictions);
modelSelect.addEventListener("change", loadPredictions);
canvas.addEventListener("mousemove", showTooltip);
canvas.addEventListener("mouseleave", () => {
  tooltip.hidden = true;
});
window.addEventListener("resize", drawChart);

init();
