"""
FastAPI bridge. Thin route over the inference pipeline; no model code here.

Run:  cd backend && uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from inference.pipeline import InferenceError, pipeline
from inference.preprocessing import PreprocessingError

ALLOWED = (".nii.gz", ".nii", ".png", ".jpg", ".jpeg")
MAX_BYTES = 25 * 1024 * 1024

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline.load()                 # load the optimized .pth ONCE at startup
    logging.info("Model '%s' loaded on %s.", pipeline.kind, pipeline.device)
    yield
    pipeline.model = None


app = FastAPI(title="CardioVision API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


def _validate(field: str, up: UploadFile, data: bytes) -> None:
    if not (up.filename or "").lower().endswith(ALLOWED):
        raise HTTPException(415, f"'{field}': unsupported file '{up.filename}'.")
    if not data:
        raise HTTPException(400, f"'{field}': empty file.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"'{field}': file too large.")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": pipeline.kind,
            "loaded": pipeline.loaded, "device": pipeline.device}


@app.post("/predict")
async def predict(
    twoch_ed: UploadFile = File(...),  twoch_es: UploadFile = File(...),
    fourch_ed: UploadFile = File(...), fourch_es: UploadFile = File(...),
) -> dict:
    fields = {"twoch_ed": twoch_ed, "twoch_es": twoch_es,
              "fourch_ed": fourch_ed, "fourch_es": fourch_es}
    images: dict[str, bytes] = {}
    for field, up in fields.items():
        data = await up.read()
        _validate(field, up, data)
        images[field] = data

    try:
        result = pipeline.predict(images)
    except PreprocessingError as e:
        raise HTTPException(415, str(e))
    except InferenceError as e:
        logging.exception("Inference failed")
        raise HTTPException(500, str(e))

    return {"success": True, "result": result.to_dict()}
