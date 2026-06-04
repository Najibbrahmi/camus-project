import { Activity, AlertOctagon, AlertTriangle, CheckCircle2, Droplet, Ruler, RotateCcw } from "lucide-react";

const CALIBRATION = {
  native:  "Calibrated from NIfTI header",
  mixed:   "Partially calibrated · some frames default spacing",
  default: "Estimated · default spacing 0.3 mm/px",
};

/**
 * Clinical Interpretation Engine.
 * EF-threshold logic per the dashboard's clinical spec:
 *   EF > 55          -> Normal systolic function
 *   45 <= EF <= 55   -> Borderline / potential early crisis
 *   EF < 45          -> Significant systolic dysfunction
 * NOTE: this is the FRONTEND interpretation used for the displayed message and
 * colour. It is intentionally self-contained so the whole card stays consistent.
 */
function interpret(ef) {
  if (ef > 55) return {
    band: "Normal", color: "#10b981", Icon: CheckCircle2,
    chip: "bg-emerald-500/10 text-emerald-300 ring-emerald-500/30",
    box:  "border-emerald-500/40 bg-emerald-500/10 text-emerald-200",
    message: "Normal systolic function. No signs of acute heart failure or crisis detected.",
  };
  if (ef >= 45) return {
    band: "Borderline", color: "#f59e0b", Icon: AlertTriangle,
    chip: "bg-amber-500/10 text-amber-300 ring-amber-500/30",
    box:  "border-amber-500/40 bg-amber-500/10 text-amber-200",
    message: "Borderline systolic function. Potential early signs of a cardiac crisis; "
           + "clinical correlation with patient symptoms is required.",
  };
  return {
    band: "Significant dysfunction", color: "#ef4444", Icon: AlertOctagon,
    chip: "bg-rose-500/10 text-rose-300 ring-rose-500/30",
    box:  "border-rose-500/40 bg-rose-500/10 text-rose-200",
    message: "Significant systolic dysfunction. High risk of heart failure or acute cardiac crisis. "
           + "Immediate clinical intervention and diagnostic workup advised.",
  };
}

function Stat({ icon: Icon, label, value }) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-slate-800 bg-slate-900/50 px-4 py-3">
      <div className="grid h-9 w-9 place-items-center rounded-lg bg-slate-950 text-sky-400 ring-1 ring-slate-800">
        <Icon className="h-4 w-4" />
      </div>
      <div>
        <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">{label}</p>
        <p className="text-sm font-semibold text-slate-100">{value}</p>
      </div>
    </div>
  );
}

export default function ResultsCard({ result, onReset }) {
  const interp = interpret(result.ef);
  const confidencePct = result.confidence != null ? Math.round(result.confidence * 100) : null;
  const fmtVol = (v) => (v != null ? `${v} mL` : "—");

  return (
    <div className="animate-fade-in-up overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/60 shadow-xl">
      <div className="h-1.5 w-full" style={{ background: interp.color }} />
      <div className="p-6">
        <div className="flex items-center gap-2 text-slate-400">
          <Activity className="h-4 w-4 text-sky-400" />
          <h3 className="text-sm font-semibold tracking-tight text-slate-100">Analysis Result</h3>
        </div>

        {/* EF value */}
        <div className="mt-4 flex items-end gap-4">
          <div>
            <p className="text-xs text-slate-500">Ejection Fraction</p>
            <p className="text-5xl font-extrabold tracking-tight" style={{ color: interp.color }}>
              {result.ef}<span className="text-2xl">%</span>
            </p>
          </div>
          <span className={`mb-1 inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold ring-1 ${interp.chip}`}>
            {interp.band}
          </span>
        </div>

        {/* Clinical Interpretation message — immediately below the EF result */}
        <div className={`mt-4 flex items-start gap-2.5 rounded-xl border px-4 py-3 text-sm leading-snug ${interp.box}`}>
          <interp.Icon className="mt-0.5 h-4 w-4 shrink-0" />
          <p>{interp.message}</p>
        </div>

        {/* Volumes */}
        <div className="mt-4 grid grid-cols-2 gap-3">
          <Stat icon={Droplet} label="EDV (biplane)" value={fmtVol(result.edv_ml)} />
          <Stat icon={Droplet} label="ESV (biplane)" value={fmtVol(result.esv_ml)} />
        </div>

        {/* Confidence + calibration transparency */}
        {confidencePct != null && (
          <div className="mt-4">
            <div className="mb-1 flex justify-between text-[11px] font-medium text-slate-500">
              <span>Model confidence</span><span>{confidencePct}%</span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-slate-800">
              <div className="h-full rounded-full bg-sky-500 transition-all duration-700" style={{ width: `${confidencePct}%` }} />
            </div>
          </div>
        )}
        {result.calibration && (
          <span className="mt-3 inline-flex items-center gap-1.5 rounded-full bg-slate-800/60 px-3 py-1 text-[11px] font-medium text-slate-300 ring-1 ring-slate-700">
            <Ruler className="h-3.5 w-3.5" /> {CALIBRATION[result.calibration]}
          </span>
        )}

        <button onClick={onReset}
          className="mt-5 inline-flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-800/60 px-4 py-2 text-sm font-semibold text-slate-200 hover:bg-slate-800">
          <RotateCcw className="h-4 w-4" /> New analysis
        </button>

        {/* Disclaimer */}
        <p className="mt-5 border-t border-slate-800 pt-4 text-[11px] leading-relaxed text-slate-500">
          Disclaimer: Automated analysis for clinical support only. Results must be verified by a
          qualified cardiologist.
        </p>
      </div>
    </div>
  );
}
