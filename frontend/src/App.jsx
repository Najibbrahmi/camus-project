import { useEffect, useMemo, useState } from "react";
import { AlertCircle, Loader2, Sparkles, User } from "lucide-react";
import Sidebar from "./components/Sidebar";
import UploadSlot from "./components/UploadSlot";
import ResultsCard from "./components/ResultsCard";
import HistoryPage from "./pages/HistoryPage";
import { SLOTS, validateSlot, hasBlockingError } from "./lib/validateSlot";
import { predictEF } from "./lib/api";
import { getHistory, writeHistory, makeId, patientIdFromFiles } from "./lib/historyStore";

export default function App() {
  // --- Local state-based navigation (no router) ---
  const [activePage, setActivePage] = useState("analysis"); // "analysis" | "history"

  // --- Central history state: lifted here so it is NOT wiped on page switch ---
  const [historyData, setHistoryData] = useState(() => getHistory());
  useEffect(() => { writeHistory(historyData); }, [historyData]); // mirror to localStorage

  // Called when an analysis completes — bundles patientName with the result.
  const addToHistory = (result, patientName) => {
    const { masks, ...slim } = result || {};   // drop base64 masks (storage size)
    setHistoryData((prev) => [
      { id: makeId(), date: new Date().toISOString(), patientName: patientName || "Unknown", ...slim },
      ...prev,
    ]);
  };

  // Called by the History page's "Add Patient" modal.
  const addManualPatient = (record) => {
    setHistoryData((prev) => [{ id: makeId(), manual: true, ...record }, ...prev]);
  };

  return (
    <div className="flex min-h-screen">
      <Sidebar active={activePage} onNavigate={setActivePage} />
      <div className="min-w-0 flex-1">
        {activePage === "analysis" ? (
          <AnalysisView addToHistory={addToHistory} />
        ) : (
          <HistoryPage
            historyData={historyData}
            onAddManual={addManualPatient}
            onClear={() => setHistoryData([])}
          />
        )}
      </div>
    </div>
  );
}

/* ----------------------------- New Analysis ----------------------------- */
function AnalysisView({ addToHistory }) {
  const [patientName, setPatientName] = useState("");
  const [files, setFiles] = useState({});
  const [validations, setValidations] = useState({});
  const [status, setStatus] = useState("idle"); // idle | processing | done | error
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const allPresent = SLOTS.every((s) => files[s.key]);
  const anyBlocking = Object.values(validations).some(hasBlockingError);
  const canAnalyze = allPresent && !anyBlocking && status !== "processing";
  const remaining = useMemo(() => SLOTS.length - Object.keys(files).length, [files]);

  const handleSelect = (slot, file) => {
    const validation = validateSlot(file, slot);          // validate the fresh file
    setFiles((p) => ({ ...p, [slot.key]: file }));
    setValidations((p) => ({ ...p, [slot.key]: validation }));
    setStatus("idle"); setResult(null);
  };
  const handleClear = (slot) => {
    setFiles((p) => { const n = { ...p }; delete n[slot.key]; return n; });
    setValidations((p) => { const n = { ...p }; delete n[slot.key]; return n; });
  };

  const handleAnalyze = async () => {
    setStatus("processing"); setError(null); setResult(null);
    try {
      const r = await predictEF(files);
      setResult(r); setStatus("done");
      // Bundle the patient name (fallback to the filename-derived ID) and push to history.
      addToHistory(r, patientName.trim() || patientIdFromFiles(files));
    } catch (e) {
      setError(e.message || "Analysis failed."); setStatus("error");
    }
  };

  return (
    <main className="mx-auto max-w-6xl px-6 py-8">
      <h2 className="text-2xl font-bold tracking-tight text-slate-100">Cardiac Function Analysis</h2>
      <p className="mt-1 text-sm text-slate-500">
        Upload the four biplane echo frames. Each slot expects a specific view and phase.
      </p>

      {/* Patient Name / ID */}
      <div className="mt-6 max-w-sm">
        <label className="mb-1.5 block text-xs font-medium text-slate-400">Patient Name / ID</label>
        <div className="relative">
          <User className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
          <input
            type="text"
            value={patientName}
            onChange={(e) => setPatientName(e.target.value)}
            placeholder="e.g. patient0451 or John Doe"
            disabled={status === "processing"}
            className="w-full rounded-xl border border-slate-700 bg-slate-900/60 py-2.5 pl-9 pr-3 text-sm text-slate-100 placeholder:text-slate-600 focus:border-sky-500/60 focus:outline-none focus:ring-2 focus:ring-sky-500/30 disabled:opacity-60"
          />
        </div>
      </div>

      <div className="mt-8 grid gap-6 lg:grid-cols-5">
        {/* Upload grid */}
        <section className="lg:col-span-3">
          <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-6">
            <div className="mb-4 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-200">Echocardiography Input</h3>
              <span className="rounded-full bg-slate-800 px-2.5 py-1 text-xs font-semibold text-slate-400">
                {Object.keys(files).length}/4
              </span>
            </div>
            <div className="grid grid-cols-2 gap-4">
              {SLOTS.map((slot) => (
                <UploadSlot
                  key={slot.key}
                  slot={slot}
                  file={files[slot.key] || null}
                  validation={validations[slot.key]}
                  maskSrc={status === "done" ? result?.masks?.[slot.key] : null}
                  onSelect={handleSelect}
                  onClear={handleClear}
                  disabled={status === "processing"}
                />
              ))}
            </div>

            <div className="mt-5 flex items-center justify-between">
              <p className="text-xs text-slate-500">
                {anyBlocking ? "Resolve the highlighted slots to continue."
                 : remaining > 0 ? `${remaining} frame${remaining === 1 ? "" : "s"} remaining`
                 : "All frames ready."}
              </p>
              <button onClick={handleAnalyze} disabled={!canAnalyze}
                className="inline-flex items-center gap-2 rounded-xl bg-sky-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm shadow-sky-600/20 transition hover:bg-sky-700 disabled:cursor-not-allowed disabled:opacity-50">
                {status === "processing"
                  ? (<><Loader2 className="h-4 w-4 animate-spin" /> Processing…</>)
                  : (<><Sparkles className="h-4 w-4" /> Analyze EF</>)}
              </button>
            </div>
          </div>
        </section>

        {/* Results panel */}
        <section className="lg:col-span-2">
          {status === "done" && result && (
            <ResultsCard result={result} onReset={() => { setStatus("idle"); setResult(null); }} />
          )}

          {status === "error" && (
            <div className="animate-fade-in-up flex items-start gap-3 rounded-2xl border border-rose-500/40 bg-rose-500/10 p-5 text-sm text-rose-300">
              <AlertCircle className="mt-0.5 h-5 w-5 shrink-0" />
              <div><p className="font-semibold">Analysis failed</p><p className="mt-0.5 text-rose-300/80">{error}</p></div>
            </div>
          )}

          {(status === "idle" || status === "processing") && (
            <div className="grid h-full min-h-[280px] place-items-center rounded-2xl border border-dashed border-slate-800 bg-slate-900/30 p-8 text-center">
              <div>
                <div className="mx-auto mb-3 grid h-12 w-12 place-items-center rounded-2xl bg-sky-500/10 text-sky-400">
                  {status === "processing" ? <Loader2 className="h-6 w-6 animate-spin" /> : <Sparkles className="h-6 w-6" />}
                </div>
                <p className="text-sm font-medium text-slate-400">
                  {status === "processing" ? "Segmenting & computing EF…" : "Results will appear here"}
                </p>
                <p className="mt-1 text-xs text-slate-600">Add all four frames, then run the analysis.</p>
              </div>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
