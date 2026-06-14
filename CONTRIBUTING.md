# Contributing

Thanks for your interest! A few guidelines specific to this project.

## Setup

```bash
pip install -e ".[dev]"
pytest          # fast, offline
ruff check src tests
```

## The one rule that matters

**A change is adopted only if it is a significant out-of-sample win.** Run it through the
validation harness (`wcforecast validate` / `wcforecast.validate`) on the locked test window
and report the paired-bootstrap CI. Do **not** justify a change by how it looks on a handful
of recent matches — see `docs/FINDINGS.md` for why that over-fits.

Two hard constraints:
- **No market data as a model input** (features, calibration target, or ensemble). Markets
  are benchmark-only.
- **No leakage**: anything used to predict a match must be computed strictly before it.

## Good contributions

- New leakage-safe strength signals (with an OOS evaluation).
- Better calibration or simulation fidelity (e.g. the exact FIFA third-place table).
- More tests, docs, and reproducibility improvements.

## Style

`ruff` (config in `pyproject.toml`), type hints and docstrings on public functions, and keep
modules focused.
