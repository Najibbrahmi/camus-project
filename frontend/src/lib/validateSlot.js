// The 4 required frames. `key` matches the FastAPI /predict field names.
export const SLOTS = [
  { key: "twoch_ed", code: "2CH-ED", label: "2-Chamber · End-Diastole", view: "2CH", phase: "ED" },
  { key: "twoch_es", code: "2CH-ES", label: "2-Chamber · End-Systole", view: "2CH", phase: "ES" },
  { key: "fourch_ed", code: "4CH-ED", label: "4-Chamber · End-Diastole", view: "4CH", phase: "ED" },
  { key: "fourch_es", code: "4CH-ES", label: "4-Chamber · End-Systole", view: "4CH", phase: "ES" },
];

// Normalize defensively (null-safe): used on BOTH extracted and required values.
const norm = (s) => (s ?? "").toString().toUpperCase().trim();

/**
 * Extract {view, phase} from a filename — NO REGEX, plain string.includes().
 * Filenames follow a predictable structure (…2CH/4CH… and …ED/ES…), so a simple
 * substring check is the most robust option. ES is checked before ED so an "ES"
 * file is never mis-tagged as "ED".
 */
export function inferTags(filename) {
  const n = (filename || "").toUpperCase();
  const view = n.includes("2CH") ? "2CH" : n.includes("4CH") ? "4CH" : null;
  const phase = n.includes("ES") ? "ES" : n.includes("ED") ? "ED" : null;
  return { view, phase };
}

/**
 * Validate a file against its slot.
 *   error -> a token was detected but mismatches (hard block — cannot analyze)
 *   warn  -> view/phase not found in filename (soft — allowed, but flagged)
 *   ok    -> filename matches the slot
 */
export function validateSlot(file, slot) {
  if (!file) return null;

  const tags = inferTags(file.name);
  const exView = norm(tags.view);   // "" if null
  const exPhase = norm(tags.phase);
  const reqView = norm(slot.view);
  const reqPhase = norm(slot.phase);
  const match = exView === reqView && exPhase === reqPhase;

  // --- STRICT DEBUG (remove before production) ------------------------------
  console.log("[validateSlot] ───────────────────────────────");
  console.log("  parsing filename :", file.name);
  console.log("  inferTags →       :", tags);
  console.log("  slot requires     :", { view: slot.view, phase: slot.phase });
  console.log("  normalized        :", { exView, exPhase, reqView, reqPhase });
  console.log("  MATCH (view&&phase):", match);
  // --------------------------------------------------------------------------

  if (tags.view && exView !== reqView)
    return { level: "error",
      message: `This looks like a ${tags.view} frame. The ${slot.code} slot requires a ${slot.view} view.` };

  if (tags.phase && exPhase !== reqPhase)
    return { level: "error",
      message: `This looks like a ${tags.view ?? ""}-${tags.phase} frame. The ${slot.code} slot requires `
             + `${slot.phase} (${slot.phase === "ED" ? "end-diastole" : "end-systole"}).` };

  if (!tags.view || !tags.phase)
    return { level: "warn",
      message: "Couldn't verify view/phase from the filename — please confirm this is correct." };

  return { level: "ok", message: "" };
}

export const hasBlockingError = (v) => v?.level === "error";
