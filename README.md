# MLOps Lab 1 — GitHub Actions CI (Modified)

**Course:** IE 7374 / MLOps
**Base lab:** `Labs/Github_Labs/Lab1` from [raminmohammadi/MLOps](https://github.com/raminmohammadi/MLOps)
**Author:** Sathyasai

This lab covers the same five modules as the original — virtual environment,
GitHub repo + folder structure, source code, pytest/unittest tests, and
GitHub Actions CI — but with the sample code replaced by a data-analytics
utility module.

## What I changed (vs. the original lab)

| Original | This repo |
|---|---|
| `src/calculator.py` with `fun1`–`fun4` (add/sub/mul) | `src/data_utils.py` — descriptive stats, min-max + z-score scaling, IQR outlier detection, CSV column loader |
| Tests on 4 arithmetic functions | Rewritten `pytest` (with `parametrize` + `pytest.approx`) and `unittest` suites covering 10+ cases, including error paths and file I/O |
| No dataset | `data/sample.csv` loaded and tested by `load_column` |
| `actions/checkout@v2`, `setup-python@v2`, `upload-artifact@v2`, Python 3.8 | Upgraded to `checkout@v4`, `setup-python@v5`, `upload-artifact@v4`, Python 3.11 (the v2 actions are deprecated and now fail on GitHub-hosted runners) |
| Push-only triggers | Triggers on `push` **and** `pull_request` to `main` |

## Module: `src/data_utils.py`

- `mean`, `median`, `variance`, `std`
- `min_max_scale`, `z_score_normalize`
- `detect_outliers_iqr` — Tukey 1.5·IQR fences
- `summary` — one-call descriptive summary dict
- `load_column(path, column)` — read a numeric CSV column as floats

## Project structure

```
.
├── .github/workflows/
│   ├── pytest_action.yml
│   └── unittest_action.yml
├── data/
│   └── sample.csv
├── src/
│   ├── __init__.py
│   └── data_utils.py
├── test/
│   ├── __init__.py
│   ├── test_pytest.py
│   └── test_unittest.py
├── conftest.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Run locally

```bash
python -m venv lab_01
source lab_01/bin/activate      # Windows: lab_01\Scripts\activate
pip install -r requirements.txt

pytest                          # pytest suite
python -m unittest test.test_unittest   # unittest suite
```

## CI

Both workflows run automatically on every push / PR to `main`. The pytest
workflow also uploads a JUnit XML report as a build artifact.
