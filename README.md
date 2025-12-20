# Airline Passenger Satisfaction App (Fall 2025)

This project is based on the starter repo **mkzia/housing_app_fall25**, updated for a **binary classification** problem using the Airline Passenger Satisfaction dataset.

## What’s inside
- **SQLite 3NF database**: `data/airline.db`
- **Notebooks** (in `notebooks/`)
  - `01_create_database.ipynb` — build the 3NF DB (passenger/trip/service/delay/satisfaction)
  - `02_train_model_without_optuna.ipynb` — SQL JOIN → EDA → train/test split → baseline model → save `models/global_best_model.pkl`
  - `03_train_models_with_optuna.ipynb` — required experiments (with/without PCA, with/without tuning) → save `models/global_best_model_optuna.pkl`
  - `04_generate_streamlit_options.ipynb` — create `data/data_schema.json` for Streamlit UI
- **FastAPI** service (`api/`) that loads `models/global_best_model_optuna.pkl` (configurable via `.env`)
- **Streamlit** UI (`streamlit/`) that calls the API
- **Docker Compose** for local + cloud deployment

---

## Local Python setup (no Docker)

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r api/requirements.txt -r streamlit/requirements.txt
pip install -r requirements.notebooks.txt
```

Run notebooks in order (from the repo root):

1. `notebooks/01_create_database.ipynb`
2. `notebooks/02_train_model_without_optuna.ipynb`
3. `notebooks/03_train_models_with_optuna.ipynb`
4. `notebooks/04_generate_streamlit_options.ipynb`

> Note: the notebooks use robust paths, so they work whether you run them from the repo root or from inside `notebooks/`.

---

## Quick start with Docker (recommended for grading)

### 1) Install Docker Desktop
On Windows, install Docker Desktop and ensure:
- Docker Desktop is running
- **WSL 2 backend** is enabled (Settings → General)
- Your distro is enabled (Settings → Resources → WSL Integration)

### 2) Run from repo root

```bash
docker compose up -d --build
```

### 3) Open
- Streamlit UI: http://localhost:8501
- API health: http://localhost:8000/health
- API docs: http://localhost:8000/docs

Stop:

```bash
docker compose down
```

---

## Environment variables
See `.env.example`. For local docker-compose, copy:

```bash
cp .env.example .env
```

The API reads:
- `MODEL_PATH` (default: `/app/models/global_best_model_optuna.pkl`)

---

## Sanity check
You can quickly verify saved models:

```bash
python test_inference.py
```
