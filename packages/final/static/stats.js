const metricsBody = document.querySelector("#metricsBody");
const statsStatus = document.querySelector("#statsStatus");
const statsSubtitle = document.querySelector("#statsSubtitle");
const bestModelNote = document.querySelector("#bestModelNote");

function formatNumber(value, digits = 4) {
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

function anomalyCount(confusionMatrix) {
  if (!Array.isArray(confusionMatrix) || confusionMatrix.length < 2) return "-";
  const falseNegative = confusionMatrix[1]?.[0] ?? 0;
  const truePositive = confusionMatrix[1]?.[1] ?? 0;
  return falseNegative + truePositive;
}

function modelScore(model) {
  return model.test?.f1 ?? -1;
}

async function initStats() {
  try {
    statsStatus.textContent = "Loading metrics...";
    const data = await fetchJson("/api/models");
    const models = [...data.models].sort((a, b) => modelScore(b) - modelScore(a));
    metricsBody.innerHTML = "";

    for (const [index, model] of models.entries()) {
      const row = document.createElement("tr");
      if (model.is_default) row.classList.add("default-model");
      row.innerHTML = `
        <td>${index + 1}</td>
        <td><strong>${model.name}</strong>${model.is_default ? " <span class=\"badge\">default</span>" : ""}</td>
        <td>${model.model_type}</td>
        <td>${formatNumber(model.test?.precision)}</td>
        <td>${formatNumber(model.test?.recall)}</td>
        <td>${formatNumber(model.test?.f1)}</td>
        <td>${formatNumber(model.threshold, 6)}</td>
        <td>${anomalyCount(model.test?.confusion_matrix)}</td>
      `;
      metricsBody.append(row);
    }

    const best = models[0];
    statsSubtitle.textContent = `${models.length} exported models ranked by test F1.`;
    bestModelNote.textContent = best
      ? `${best.name} is currently the strongest exported model by test F1 (${formatNumber(best.test?.f1)}).`
      : "No exported model metrics were found.";
    statsStatus.textContent = "Ready.";
  } catch (error) {
    statsStatus.textContent = error.message;
  }
}

initStats();
