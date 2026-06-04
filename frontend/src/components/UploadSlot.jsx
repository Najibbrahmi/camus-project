import { useEffect, useRef, useState } from "react";
import { FileCheck2, ImageUp, X } from "lucide-react";

export default function UploadSlot({ slot, file, validation, maskSrc, onSelect, onClear, disabled }) {
  const ref = useRef(null);
  const [drag, setDrag] = useState(false);
  const [preview, setPreview] = useState(null);
  const err = validation?.level === "error";
  const warn = validation?.level === "warn";

  useEffect(() => {
    if (file && /\.(png|jpe?g)$/i.test(file.name)) {
      const url = URL.createObjectURL(file);
      setPreview(url);
      return () => URL.revokeObjectURL(url);
    }
    setPreview(null);
  }, [file]);

  const take = (list) => { if (!disabled && list?.[0]) onSelect(slot, list[0]); };

  return (
    <div className="flex flex-col">
      <div
        onClick={() => !disabled && ref.current?.click()}
        onDragOver={(e) => { e.preventDefault(); if (!disabled) setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); take(e.dataTransfer.files); }}
        className={[
          "group relative flex aspect-square cursor-pointer flex-col items-center justify-center",
          "rounded-2xl border-2 border-dashed p-4 text-center transition-all duration-200",
          drag ? "scale-[1.02] border-sky-400 bg-sky-500/10"
          : err ? "border-rose-500/70 bg-rose-500/5"
          : file ? "border-emerald-400/60 bg-emerald-400/5"
          : "border-slate-700 bg-slate-900/40 hover:border-sky-500/60 hover:bg-sky-500/5",
          disabled ? "pointer-events-none opacity-60" : "",
        ].join(" ")}
      >
        <span className="absolute left-3 top-3 rounded-md bg-slate-950/70 px-2 py-0.5 text-[11px] font-bold tracking-wide text-sky-300 ring-1 ring-slate-700">
          {slot.code}
        </span>
        {file && (
          <button type="button" aria-label={`Remove ${slot.code}`}
            onClick={(e) => { e.stopPropagation(); onClear(slot); }}
            className="absolute right-2 top-2 z-10 grid h-6 w-6 place-items-center rounded-full bg-slate-800 text-slate-400 ring-1 ring-slate-700 hover:text-rose-400">
            <X className="h-3.5 w-3.5" />
          </button>
        )}

        {/* prediction overlay (after analysis) on top of the preview */}
        {preview ? (
          <div className="relative h-20 w-20">
            <img src={preview} alt={slot.code} className="h-20 w-20 rounded-lg object-cover ring-1 ring-slate-700" />
            {maskSrc && <img src={maskSrc} alt="" className="absolute inset-0 h-20 w-20 rounded-lg object-cover" />}
          </div>
        ) : file ? (
          <FileCheck2 className="h-8 w-8 text-emerald-400" />
        ) : (
          <ImageUp className="h-8 w-8 text-slate-500 transition-colors group-hover:text-sky-400" />
        )}

        <p className="mt-2 text-[11px] font-medium text-slate-400">{slot.label}</p>
        <p className="mt-0.5 max-w-full truncate px-1 text-[10px] text-slate-500">
          {file ? file.name : "Drop or click"}
        </p>
        <input ref={ref} type="file" accept=".nii.gz,.nii,.png,.jpg,.jpeg" className="hidden"
               onChange={(e) => take(e.target.files)} />
      </div>

      {/* PERSISTENT per-slot message */}
      {validation && validation.level !== "ok" && (
        <p className={`mt-1.5 text-[11px] leading-snug ${err ? "text-rose-400" : "text-amber-400"}`}>
          {err ? "⛔ " : "⚠️ "}{validation.message}
        </p>
      )}
    </div>
  );
}
