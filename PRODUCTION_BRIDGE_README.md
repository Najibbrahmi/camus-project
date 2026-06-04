# CardioVision — Production Bridge

Web front-end + FastAPI bridge over your **existing, optimized** CAMUS models.
The bridge imports inference-only copies of your architectures and loads your
`.pth` weights — it never imports or modifies the training notebooks.

```
camus_project/
├── 01..04_*.ipynb              # training — untouched
├── best_model.pth, mae_pretrained_full.pth, results/…   # weights (used read-only)
├── backend/
│   ├── main.py                 # FastAPI: /predict, /health, CORS
│   ├── requirements.txt
│   └── inference/
│       ├── models.py           # UNet + ViTSegmentationModel (inference copies)
│       ├── preprocessing.py    # bytes -> tensor + NIfTI spacing
│       ├── simpson.py          # biplane Simpson -> EF/volumes
│       └── pipeline.py         # loads .pth ONCE -> segment -> EF + masks
└── frontend/                   # React + Vite + Tailwind (dark-blue)
    └── src/
        ├── App.jsx             # idle -> processing -> done state machine
        ├── lib/{validateSlot.js, api.js}
        └── components/{UploadSlot.jsx, ResultsCard.jsx}
```

## Run (two terminals)

**Backend**
```powershell
cd backend
python -m venv .venv ; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 8000      # serves U-Net (best_model.pth) by default
```
Switch to the MAE+ViT model:  `$env:MODEL_KIND="vit"; uvicorn main:app --reload --port 8000`

**Frontend**
```powershell
cd frontend
npm install
npm run dev          # http://localhost:5173
```

## Contract
`POST /predict` — multipart fields `twoch_ed, twoch_es, fourch_ed, fourch_es` ->
```json
{ "success": true, "result": {
  "ef": 21.2, "edv_ml": 111.3, "esv_ml": 87.6,
  "label": "Severely reduced", "severity": "severe",
  "confidence": 1.0, "calibration": "native",
  "masks": { "twoch_ed": "data:image/png;base64,…", … } } }
```

## Validation (frontend)
Filename-based per slot: a clear mismatch (e.g. a `…_4CH_ES…` file in the `2CH-ED`
slot) **hard-blocks** Analyze with a persistent error; a filename with no view/phase
token gives a **soft warning** (allowed). Pixel-level view detection is intentionally
NOT claimed — it would need a server-side classifier.
```
