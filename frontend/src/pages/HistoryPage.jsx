import { useMemo, useState } from "react";
import { FileText, Trash2, X, Inbox, Search, UserPlus } from "lucide-react";
import { statusForRow, statusFromEf } from "../lib/historyStore";
import ResultsCard from "../components/ResultsCard";

function fmtDate(iso) {
  try {
    return new Date(iso).toLocaleString(undefined, {
      year: "numeric", month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

export default function HistoryPage({ historyData = [], onAddManual, onClear }) {
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState(null);   // entry shown in the report modal
  const [showAdd, setShowAdd] = useState(false);

  // Real-time filtering by patient name.
  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return historyData;
    return historyData.filter((r) => (r.patientName || "").toLowerCase().includes(q));
  }, [historyData, query]);

  return (
    <main className="mx-auto max-w-6xl px-6 py-8">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-slate-100">Patient History</h2>
          <p className="mt-1 text-sm text-slate-500">Previously analyzed sessions, most recent first.</p>
        </div>

        {/* Search (top-right) + actions */}
        <div className="flex items-center gap-2">
          <div className="relative">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search patient name…"
              className="w-56 rounded-xl border border-slate-700 bg-slate-900/60 py-2 pl-9 pr-3 text-sm text-slate-100 placeholder:text-slate-600 focus:border-sky-500/60 focus:outline-none focus:ring-2 focus:ring-sky-500/30"
            />
          </div>
          <button onClick={() => setShowAdd(true)}
            className="inline-flex items-center gap-2 rounded-xl bg-sky-600 px-3 py-2 text-xs font-semibold text-white hover:bg-sky-700">
            <UserPlus className="h-3.5 w-3.5" /> Add Patient
          </button>
          {historyData.length > 0 && (
            <button onClick={() => { if (window.confirm("Clear all saved records?")) onClear?.(); }}
              className="inline-flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-800/60 px-3 py-2 text-xs font-semibold text-slate-300 hover:bg-slate-800">
              <Trash2 className="h-3.5 w-3.5" /> Clear
            </button>
          )}
        </div>
      </div>

      {rows.length === 0 ? (
        <div className="mt-8 grid place-items-center rounded-2xl border border-dashed border-slate-800 bg-slate-900/30 p-16 text-center">
          <div>
            <div className="mx-auto mb-3 grid h-12 w-12 place-items-center rounded-2xl bg-slate-800 text-slate-500">
              <Inbox className="h-6 w-6" />
            </div>
            <p className="text-sm font-medium text-slate-400">
              {query ? "No patients match your search." : "No analyses yet"}
            </p>
            {!query && (
              <p className="mt-1 text-xs text-slate-600">Run an analysis, or add a patient manually — records appear here.</p>
            )}
          </div>
        </div>
      ) : (
        <div className="mt-8 overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/40">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-slate-800 bg-slate-900/60 text-[11px] uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-5 py-3 font-semibold">Patient Name</th>
                <th className="px-5 py-3 font-semibold">Date</th>
                <th className="px-5 py-3 font-semibold">EF Value</th>
                <th className="px-5 py-3 font-semibold">Status</th>
                <th className="px-5 py-3 text-right font-semibold">Report</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70">
              {rows.map((r) => {
                const st = statusForRow(r);
                return (
                  <tr key={r.id} className="transition hover:bg-slate-800/30">
                    <td className="px-5 py-3 font-medium text-slate-200">{r.patientName}</td>
                    <td className="px-5 py-3 text-slate-400">{fmtDate(r.date)}</td>
                    <td className="px-5 py-3 font-semibold" style={{ color: st.dot }}>
                      {r.ef != null ? `${r.ef}%` : "—"}
                    </td>
                    <td className="px-5 py-3">
                      <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold ring-1 ${st.chip}`}>
                        <span className="h-1.5 w-1.5 rounded-full" style={{ background: st.dot }} />
                        {st.label}
                      </span>
                    </td>
                    <td className="px-5 py-3 text-right">
                      <button onClick={() => setSelected(r)} disabled={r.ef == null}
                        className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800/60 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40">
                        <FileText className="h-3.5 w-3.5" /> View Report
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Report modal — reuses the live dashboard's ResultsCard */}
      {selected && (
        <Overlay onClose={() => setSelected(null)}>
          <div className="mb-2 flex items-center justify-between px-1">
            <p className="text-sm font-semibold text-slate-200">{selected.patientName} · {fmtDate(selected.date)}</p>
            <button onClick={() => setSelected(null)} className="text-slate-400 hover:text-slate-200"><X className="h-5 w-5" /></button>
          </div>
          <ResultsCard result={selected} onReset={() => setSelected(null)} />
        </Overlay>
      )}

      {/* Add Patient modal */}
      {showAdd && (
        <AddPatientModal
          onClose={() => setShowAdd(false)}
          onSubmit={(rec) => { onAddManual?.(rec); setShowAdd(false); }}
        />
      )}
    </main>
  );
}

/* ------------------------------ Modal shell ----------------------------- */
function Overlay({ children, onClose }) {
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-slate-950/70 p-4 backdrop-blur-sm" onClick={onClose}>
      <div className="w-full max-w-md" onClick={(e) => e.stopPropagation()}>{children}</div>
    </div>
  );
}

/* --------------------------- Add Patient form --------------------------- */
function AddPatientModal({ onClose, onSubmit }) {
  const today = new Date().toISOString().slice(0, 10);
  const [name, setName] = useState("");
  const [date, setDate] = useState(today);
  const [ef, setEf] = useState("");

  const efNum = ef === "" ? null : Number(ef);
  const st = efNum != null && !Number.isNaN(efNum) ? statusFromEf(efNum) : null;
  const valid = name.trim() && efNum != null && !Number.isNaN(efNum) && efNum >= 0 && efNum <= 100;

  const submit = () => {
    if (!valid) return;
    onSubmit({
      patientName: name.trim(),
      date: new Date(date).toISOString(),
      ef: efNum,
      // manual record: no model-derived fields
      edv_ml: null, esv_ml: null, confidence: null, calibration: null,
    });
  };

  const field = "w-full rounded-xl border border-slate-700 bg-slate-900/60 px-3 py-2.5 text-sm text-slate-100 placeholder:text-slate-600 focus:border-sky-500/60 focus:outline-none focus:ring-2 focus:ring-sky-500/30";

  return (
    <Overlay onClose={onClose}>
      <div className="rounded-2xl border border-slate-800 bg-slate-900/90 p-6 shadow-xl">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-base font-semibold text-slate-100">Add Patient Record</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-200"><X className="h-5 w-5" /></button>
        </div>

        <div className="space-y-3">
          <div>
            <label className="mb-1 block text-xs font-medium text-slate-400">Patient Name / ID</label>
            <input className={field} value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. John Doe" />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="mb-1 block text-xs font-medium text-slate-400">Date</label>
              <input type="date" className={field} value={date} onChange={(e) => setDate(e.target.value)} />
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-slate-400">EF (%)</label>
              <input type="number" min="0" max="100" className={field} value={ef} onChange={(e) => setEf(e.target.value)} placeholder="0–100" />
            </div>
          </div>

          {/* Live status (auto-derived from EF so it can't contradict the value) */}
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <span>Status:</span>
            {st ? (
              <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 font-semibold ring-1 ${st.chip}`}>
                <span className="h-1.5 w-1.5 rounded-full" style={{ background: st.dot }} />{st.label}
              </span>
            ) : (
              <span className="text-slate-600">enter an EF value…</span>
            )}
          </div>
        </div>

        <div className="mt-6 flex justify-end gap-2">
          <button onClick={onClose} className="rounded-xl border border-slate-700 bg-slate-800/60 px-4 py-2 text-sm font-semibold text-slate-300 hover:bg-slate-800">Cancel</button>
          <button onClick={submit} disabled={!valid}
            className="rounded-xl bg-sky-600 px-4 py-2 text-sm font-semibold text-white hover:bg-sky-700 disabled:cursor-not-allowed disabled:opacity-50">
            Add Record
          </button>
        </div>
      </div>
    </Overlay>
  );
}
