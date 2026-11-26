"use strict";

const MAX_CHARACTERS = 10000;
const HISTORY_KEY = "clauseforge-session-history-v1";
const THEME_KEY = "clauseforge-theme-v1";
const samples = [
  ["Governing law", "This Agreement is governed by the laws of the State of Delaware, without regard to conflict-of-law principles."],
  ["Termination", "Either party may terminate this Agreement for convenience upon sixty days' written notice to the other party."],
  ["Non-compete", "During the term, the Distributor will not market products that directly compete with the Licensed Products in the Territory."],
  ["Audit rights", "The Company may inspect and audit the relevant books and records once per calendar year upon reasonable prior notice."],
  ["Change of control", "A change in control of either party requires prompt written notice to the other party."],
];

const byId = (id) => document.getElementById(id);
const elements = {
  text: byId("clause-text"), count: byId("character-count"), analyze: byId("analyze-button"), clear: byId("clear-button"),
  error: byId("form-error"), empty: byId("result-empty"), loading: byId("result-loading"), success: byId("result-success"), resultError: byId("result-error"),
};
let taxonomyByCanonical = new Map();
let sampleReplays;
const replayReady = fetch("/static/sample-replays.json").then(r => r.json()).then(data => {sampleReplays=data;});

function safeJson(response) { return response.json().catch(() => ({})); }
function setVisible(target) { [elements.empty, elements.loading, elements.success, elements.resultError].forEach((node) => { node.hidden = node !== target; }); }
function formatCategory(canonical) {
  const known = taxonomyByCanonical.get(canonical);
  if (known) return known;
  const quoted = canonical.match(/[“"]([^”"]+)[”"]/);
  return { category_id: "taxonomy-category", category_name: quoted ? quoted[1] : "CUAD category", canonical };
}
function providerLabel(provider, backend, modelId) { return backend === "mock" || provider === "mock-development" ? "Development / Mock Model" : modelId ? `${modelId.split("/").pop()} + LoRA` : provider || backend || "Configured model"; }
function modelStateLabel(provider, backend) { return backend === "mock" || provider === "mock-development" ? "Mock · demonstration only" : "Active trained model candidate"; }

function updateInput() {
  const length = elements.text.value.length;
  elements.count.textContent = `${length.toLocaleString()} / ${MAX_CHARACTERS.toLocaleString()}`;
  elements.analyze.disabled = !elements.text.value.trim() || length > MAX_CHARACTERS;
  elements.error.hidden = true;
}

async function analyze() {
  const text = elements.text.value.trim();
  if (!text) { elements.error.textContent = "Enter a contract clause before analyzing."; elements.error.hidden = false; elements.text.focus(); return; }
  elements.analyze.disabled = true; elements.analyze.classList.add("loading"); elements.analyze.setAttribute("aria-busy", "true"); setVisible(elements.loading);
  try {
    await replayReady;
    const replay = sampleReplays.samples.find(row => row.text === text);
    if (!replay) throw {title:"Choose a recorded example",message:"This public demo replays five exact sample inputs. Select a sample to inspect its real recorded output; free-form inference is not hosted."};
    if (!taxonomyByCanonical.size) { const t = await fetch("/v1/taxonomy").then(r=>r.json()); taxonomyByCanonical = new Map(t.categories.map(item=>[item.canonical,item])); }
    const id = replay.raw_output.trim();
    const known = [...taxonomyByCanonical.values()].find(item => item.category_id === id);
    if (!known) throw {title:"Replayed invalid model output",message:`The real run returned: ${replay.raw_output}. This is not a valid exact category ID. No replacement prediction was invented.`};
    const data = {predicted_category:known.canonical};
    const category = formatCategory(data.predicted_category);
    byId("category-name").textContent = category.category_name;
    byId("category-id").textContent = category.category_id;
    byId("category-question").textContent = category.canonical;
    byId("result-provider").textContent = "Qwen2.5-1.5B + LoRA · replayed from real run";
    byId("result-model-state").textContent = "Replayed from real run · no live inference";
    byId("result-latency").textContent = `${(replay.duration_seconds * 1000).toFixed(1)} ms (recorded)`;
    byId("result-request").textContent = replay.label;
    setVisible(elements.success); addHistory(text, category.category_name);
  } catch (error) {
    byId("error-title").textContent = error.title || "Analysis unavailable";
    byId("error-message").textContent = error.message || "An unexpected error occurred. Please try again.";
    setVisible(elements.resultError);
  } finally { elements.analyze.classList.remove("loading"); elements.analyze.removeAttribute("aria-busy"); updateInput(); }
}

function apiError(status, data) {
  const code = data?.error?.code;
  if (status === 422) return { title: "Check the clause text", message: "The submitted text could not be analyzed. Enter one non-empty contract clause." };
  if (status === 503 || code === "provider_unavailable") return { title: "Model not ready", message: "The model backend is currently unavailable. Check System status and try again later." };
  if (code === "invalid_model_output") return { title: "Classification unavailable", message: "The model did not return a valid taxonomy category. No result was accepted." };
  return { title: "Analysis unavailable", message: "ClauseForge could not complete this request. Please try again." };
}

async function refreshStatus() {
  const pill = byId("header-status"); pill.className = "status-pill checking"; pill.lastElementChild.textContent = "Checking system";
  try {
    const [readyResponse, versionResponse, taxonomyResponse] = await Promise.all([fetch("/ready"), fetch("/version"), fetch("/v1/taxonomy")]);
    const ready = await safeJson(readyResponse); const version = await safeJson(versionResponse); const taxonomy = await safeJson(taxonomyResponse);
    if (Array.isArray(taxonomy.categories)) taxonomyByCanonical = new Map(taxonomy.categories.map((item) => [item.canonical, item]));
    setStatus("status-application", ready.application_ready, "Ready", "Unavailable");
    byId("status-artifact").textContent = !ready.model_artifact_configured ? "Not configured" : ready.model_artifact_valid ? "Valid" : "Invalid artifact";
    setStatus("status-backend", ready.model_backend_ready, "Ready", "Unavailable");
    byId("detail-provider").textContent = ready.provider || version.provider || "Not supplied"; byId("detail-backend").textContent = version.backend || "Not supplied"; byId("detail-version").textContent = version.application_version || "Not supplied"; byId("detail-build").textContent = version.build_commit || "Not supplied";
    const mock = version.backend === "mock" || ready.provider === "mock-development";
    await replayReady;
    byId("status-artifact").textContent="5 recorded real outputs";
    byId("status-backend").textContent="Replay only";
    byId("detail-backend").textContent="Saved local CPU run; the mock API is not called by Analyze";
    const notice = byId("model-notice"); notice.hidden = false;
    byId("model-notice-title").textContent = mock ? "Replayed from real run" : ready.model_backend_ready ? "Active trained model" : "Real model unavailable";
    byId("model-notice-detail").textContent = mock ? "Five exact examples use saved Qwen2.5-1.5B + LoRA outputs. No live model runs here." : ready.model_backend_ready ? `Checkpoint ${ready.checkpoint_step} · ${ready.candidate_status}. Final model selection and release validation are pending.` : "Configured real-model mode failed closed. No mock fallback is active.";
    pill.className = ready.model_backend_ready ? "status-pill" : "status-pill unavailable"; pill.lastElementChild.textContent = mock ? "Real-run replay ready" : ready.model_backend_ready ? "Trained candidate ready" : "Backend unavailable";
  } catch (_) {
    ["status-application", "status-artifact", "status-backend"].forEach((id) => { byId(id).textContent = "Unavailable"; }); pill.className = "status-pill unavailable"; pill.lastElementChild.textContent = "System unavailable";
  }
}
function setStatus(id, value, yes, no) { byId(id).textContent = value ? yes : no; }

function history() { try { return JSON.parse(sessionStorage.getItem(HISTORY_KEY) || "[]"); } catch (_) { return []; } }
function addHistory(text, category) { const items = history(); items.unshift({ preview: text.slice(0, 92), category, timestamp: new Date().toISOString() }); sessionStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, 5))); renderHistory(); }
function renderHistory() { const list = byId("history-list"); const items = history(); list.replaceChildren(); byId("history-empty").hidden = items.length > 0; items.forEach((item) => { const li = document.createElement("li"); li.className = "history-item"; const strong = document.createElement("strong"); strong.textContent = item.category; const preview = document.createElement("p"); preview.textContent = item.preview; const time = document.createElement("time"); time.dateTime = item.timestamp; time.textContent = new Date(item.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); li.append(strong, preview, time); list.append(li); }); }

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const toggle = byId("theme-toggle");
  if (toggle) toggle.setAttribute("aria-label", `Switch to ${theme === "dark" ? "light" : "dark"} theme`);
}
function initialTheme() {
  const stored = localStorage.getItem(THEME_KEY);
  return stored === "light" || stored === "dark" ? stored : window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function initialize() {
  samples.forEach(([label, text]) => { const button = document.createElement("button"); button.type = "button"; button.className = "sample-chip"; button.textContent = label; button.addEventListener("click", () => { document.querySelectorAll(".sample-chip").forEach((chip) => chip.classList.remove("selected")); button.classList.add("selected"); elements.text.value = text; updateInput(); elements.text.focus(); }); byId("sample-list").append(button); });
  elements.text.addEventListener("input", updateInput); elements.analyze.addEventListener("click", analyze); elements.clear.addEventListener("click", () => { elements.text.value = ""; updateInput(); setVisible(elements.empty); elements.text.focus(); });
  byId("text-file").addEventListener("change", async (event) => { const file = event.target.files?.[0]; if (!file) return; if (file.size > 100000) { elements.error.textContent = "Text files must be 100 KB or smaller."; elements.error.hidden = false; return; } elements.text.value = (await file.text()).slice(0, MAX_CHARACTERS); updateInput(); });
  byId("refresh-status").addEventListener("click", refreshStatus); byId("clear-history").addEventListener("click", () => { sessionStorage.removeItem(HISTORY_KEY); renderHistory(); });
  applyTheme(initialTheme());
  byId("theme-toggle").addEventListener("click", () => { const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; localStorage.setItem(THEME_KEY, theme); applyTheme(theme); });
  updateInput(); renderHistory(); refreshStatus();
}
document.addEventListener("DOMContentLoaded", initialize);

async function loadEvaluationEvidence() {
  try {
    const response = await fetch('/static/evaluation.json');
    if (!response.ok) return;
    const evidence = await response.json();
    const status = document.getElementById('evaluation-status');
    if (!status) return;
    status.textContent = 'Recorded runs on 200 held-out clauses. *Gemini known usage: 492,297 input + 807 output tokens; paid-rate equivalent ≥$0.1497 total, ≥$0.749 per 1,000 attempted clauses. Fourteen failed calls omit usage, so this is a lower bound. Actual free-tier charge: $0. Google rates checked 21 Nov 2026: $0.30 / $2.50 per million input/output tokens.';
    if (!evidence.runs?.length) return;
    const body = document.getElementById('evaluation-results');
    body.replaceChildren();
    for (const run of evidence.runs) {
      const row = document.createElement('tr');
      const names = ['Qwen2.5-1.5B · Q4 base','Qwen2.5-1.5B + LoRA · FP16 pilot','Qwen2.5-1.5B + LoRA · Q4 serving','Gemini 3.5 Flash-Lite','Qwen2.5-1.5B + LoRA · Q4, constrained','Qwen2.5-3B · Q4 base, constrained','Qwen2.5-1.5B + LoRA · 100 steps, Q4 constrained'];
      const index = evidence.runs.indexOf(run);
      const cost = index === 3 ? '$0 free tier; ≥$0.749 / 1k paid-rate equivalent*' : index === 1 ? 'Colab T4 · free GPU' : 'Local CPU';
      const values = [names[index] || run.model, run.metrics.sample_size, `${(run.metrics.accuracy * 100).toFixed(1)}%`, run.metrics.macro_f1_all_categories.toFixed(3), `${(Math.max(0,run.metrics.invalid_output_rate-run.metrics.request_error_rate)*100).toFixed(1)}%${run.metrics.request_error_rate ? ` (+${(run.metrics.request_error_rate*100).toFixed(1)}% request errors)` : ""}`, `${run.latency_median_seconds.toFixed(3)} s`, cost];
      for (const value of values) { const cell = document.createElement('td'); cell.textContent = String(value); row.append(cell); }
      body.append(row);
    }
  } catch (_) { /* Preserve explicit pending states when evidence is unavailable. */ }
}
document.addEventListener('DOMContentLoaded', loadEvaluationEvidence);

