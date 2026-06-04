import { SLOTS } from "./validateSlot";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

/**
 * POST the 4 frames to /predict.
 * @param {Record<string, File>} files keyed by slot.key
 * @returns {Promise<object>} result { ef, confidence, label, severity, edv_ml, esv_ml, calibration, masks }
 */
export async function predictEF(files) {
  const form = new FormData();
  for (const slot of SLOTS) form.append(slot.key, files[slot.key]); // fixed order

  let res;
  try {
    res = await fetch(`${API_URL}/predict`, { method: "POST", body: form });
  } catch {
    throw new Error("Cannot reach the analysis server. Is the backend running on :8000?");
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try { detail = (await res.json()).detail || detail; } catch { /* non-JSON */ }
    throw new Error(detail);
  }
  return (await res.json()).result;
}
