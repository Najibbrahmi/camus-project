// Mock persistence for the demo. The central historyData array lives in App.jsx
// state (so it survives page switches); these helpers just read/write localStorage
// so it also survives a full page reload. The FastAPI backend is untouched.

const KEY = "astrocardia_history_v1";

export function getHistory() {
  try {
    return JSON.parse(localStorage.getItem(KEY)) || [];
  } catch {
    return [];
  }
}

export function writeHistory(arr) {
  try {
    localStorage.setItem(KEY, JSON.stringify(arr));
  } catch {
    /* storage full / unavailable — ignore for the demo */
  }
}

export function makeId() {
  return (typeof crypto !== "undefined" && crypto.randomUUID)
    ? crypto.randomUUID()
    : String(Date.now()) + Math.random().toString(16).slice(2);
}

// Badge styling per clinical status.
const STYLES = {
  Healthy:    { chip: "bg-emerald-500/10 text-emerald-300 ring-emerald-500/30", dot: "#10b981" },
  Borderline: { chip: "bg-amber-500/10 text-amber-300 ring-amber-500/30",       dot: "#f59e0b" },
  Risk:       { chip: "bg-rose-500/10 text-rose-300 ring-rose-500/30",          dot: "#ef4444" },
};

export function statusByLabel(label) {
  return { label, ...(STYLES[label] || STYLES.Risk) };
}

// EF -> clinical status (Healthy / Borderline / Risk), thresholds >55 / 45–55 / <45.
export function statusFromEf(ef) {
  const label = ef > 55 ? "Healthy" : ef >= 45 ? "Borderline" : "Risk";
  return statusByLabel(label);
}

// Resolve the status for any row (computed records use EF; manual ones may store a label).
export function statusForRow(row) {
  if (row.ef != null) return statusFromEf(row.ef);
  return statusByLabel(row.status || "Risk");
}

// Derive a fallback patient ID from the uploaded filenames (patient0001-2CH-ED.png -> "patient0001").
export function patientIdFromFiles(files) {
  const f = Object.values(files).find(Boolean);
  if (!f) return "Unknown";
  const base = f.name.replace(/\.(nii\.gz|nii|png|jpe?g)$/i, "");
  const id = [];
  for (const t of base.split(/[-_.\s]+/)) {
    if (/^(2CH|4CH|ED|ES)$/i.test(t)) break;
    id.push(t);
  }
  return id.join("-") || "Unknown";
}
