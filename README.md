# 360 Image Auto-Enhance — Research Benchmark

Classical, deterministic enhancement methods for equirectangular 360° images.
See [`plan.md`](plan.md) for the full research plan.

**Locked recommendation (blind votes + runtime):** `clahe`  
(see `config/default.yaml` → `recommendation`)

## Setup

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Data

`data/zips/`, `data/frames/`, and `outputs/` are **not** in git (too large). Place zip archives in `data/zips/`, then:

```bash
python -c "from pathlib import Path; from enhance.io import ingest; ingest(Path('data/zips'), Path('data/frames'), Path('data/manifest.csv'))"
```

## Run

```bash
# Batch (all datasets × methods)
python -m enhance.run --config config/default.yaml

# Metrics / report
python -m enhance.benchmark --run outputs/<run_id>
python -m enhance.report --run outputs/<run_id>

# UI
streamlit run app/app.py -- --run outputs/<run_id>
```

CLAHE comparison report helper:

```bash
python scripts/generate_clahe_html_report.py --run outputs/<run_id> --datasets "Basement - 1-1545dc10"
```

## Tests

```bash
pytest tests/ -q
```
