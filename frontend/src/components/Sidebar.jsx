import { HeartPulse, LayoutDashboard, History } from "lucide-react";

const item = (active) =>
  `flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition ${
    active
      ? "bg-sky-600/15 text-sky-300 ring-1 ring-sky-500/30"
      : "text-slate-400 hover:bg-slate-800/60 hover:text-slate-200"
  }`;

/** Persistent sidebar. State-based navigation via props (no router). */
export default function Sidebar({ active, onNavigate }) {
  return (
    <aside className="sticky top-0 flex h-screen w-60 shrink-0 flex-col border-r border-slate-800/70 bg-slate-950/60 px-4 py-5 backdrop-blur-md">
      <div className="mb-8 flex items-center gap-3 px-1">
        <div className="grid h-10 w-10 place-items-center rounded-xl bg-sky-600 text-white shadow-lg shadow-sky-600/30">
          <HeartPulse className="h-5 w-5" />
        </div>
        <div>
          <p className="text-base font-bold tracking-tight text-slate-100">AstroCardia</p>
          <p className="text-[11px] text-slate-500">EF Analysis Suite</p>
        </div>
      </div>

      <nav className="flex flex-col gap-1">
        <button type="button" onClick={() => onNavigate("analysis")} className={item(active === "analysis")}>
          <LayoutDashboard className="h-4 w-4" /> New Analysis
        </button>
        <button type="button" onClick={() => onNavigate("history")} className={item(active === "history")}>
          <History className="h-4 w-4" /> Patient History
        </button>
      </nav>

      <p className="mt-auto px-1 text-[10px] leading-relaxed text-slate-600">
        Research use only. Not for clinical diagnosis.
      </p>
    </aside>
  );
}
