# Agriculture & Remote Sensing

Spectral readings are the observations. Crop growth stage is the hidden state. This project infers the stage from NDVI, EVI, and NDRE so a field can be followed without walking it.

The model is a left-to-right hidden Markov model:

- **Hidden states** are phenology stages (bare soil through senescence). A stage usually persists, sometimes advances, and rarely skips.
- **Observations** are a diagonal Gaussian over the three indices. NDRE is included because it often drops while NDVI is still high, which separates flowering from grain fill.
- **Live monitoring** is the forward filter: the belief on a date uses only passes that had arrived by then.
- **Season reconstruction** is the Viterbi path through the whole series.
- **Learning** is Baum-Welch on unlabeled passes, anchored to a textbook spectral signature so one short series cannot invent a backward season.

Textbook signatures are agronomic defaults, not a sensor-calibrated national model. The simulator is there so the path can be checked. An uploaded CSV is scored the same way, without that check.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000

The same model from the terminal:

```bash
python -m agri_rs simulate --crop maize --seed 7
python -m agri_rs simulate --crop maize --seed 7 --learn
python -m agri_rs infer passes.csv --crop maize
```

CSV columns: `date,ndvi` and optional `evi,ndre`. Dates are `YYYY-MM-DD`. A maize season is in `data/sample_maize.csv`.

## Tests

```bash
python3 -m tests
```

## Deploy

The app is packaged with `Dockerfile` and `fly.toml`. From this directory:

```bash
fly deploy
```
