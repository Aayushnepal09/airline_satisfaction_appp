# ✈️ Airline Passenger Satisfaction (Classification) — FastAPI + Streamlit + Docker

## Overview
This project is an end-to-end **machine learning classification system** that predicts whether an airline passenger is **Satisfied** or **Neutral/Dissatisfied** based on service ratings, travel details, and passenger information.

It includes:
- A complete ML pipeline (data → preprocessing → model → evaluation)
- A **FastAPI** backend for inference (`/predict`)
- A **Streamlit** frontend for user-friendly predictions
- **Docker + docker-compose** for reproducible deployment
- Experiment tracking using **MLflow on DagsHub**

---

## What the Project Does
Users enter passenger details (age, flight distance, delays) and service ratings (1–5 scale such as cleanliness, seat comfort, etc.).  
The system returns:
- A simple **Yes/No style output** (Satisfied or Not Satisfied)
- Optional “Show Details” section for input + raw JSON (useful for grading/demo)

---

## Tech Stack
- **Python** (data + modeling)
- **scikit-learn** (pipelines, preprocessing, classification models)
- **FastAPI** (model inference API)
- **Streamlit** (frontend UI)
- **SQLite** (normalized database used as a source for Pandas DataFrames)
- **Docker + Docker Compose** (deployment & orchestration)
- **DagsHub + MLflow** (experiment logging & tracking)

---

## Architecture (High Level)
1. **Database Layer (SQLite):** stores the dataset in a normalized structure  
2. **Training Layer:** builds preprocessing + model pipelines and evaluates performance  
3. **Experiment Tracking:** logs runs + F1-scores to MLflow (DagsHub)  
4. **Inference API (FastAPI):** loads the saved best model and serves predictions  
5. **Frontend (Streamlit):** collects user inputs and calls the FastAPI endpoint

---

## Model & Experiments
The project runs multiple experiments using different:
- classification algorithms
- PCA usage (with/without)
- hyperparameter tuning (with/without Optuna)

Each experiment records **F1-score** and saves:
- the trained model artifact
- metrics JSON
- run metadata in MLflow on DagsHub

---

## Deployment
The system is containerized using Docker and orchestrated using **docker-compose**:
- `api` service → FastAPI backend on port **8000**
- `streamlit` service → UI frontend on port **8501**

The same setup can run locally or on a cloud VM (e.g., DigitalOcean).

---

## Repository Structure
- `api/` → FastAPI inference service  
- `streamlit/` → Streamlit UI  
- `models/` → saved models + metrics artifacts  
- `data/` → dataset + schema + sqlite database  
- `notebooks/` → database creation, training, experiment scripts  
- `docker-compose.yml` → orchestrates API + frontend

---

## Demo
This wont work now as i've removed it from digitalocean if you want you can clone the project and run it locally 
- Streamlit UI: `http://<host>:8501`
- FastAPI docs: `http://<host>:8000/docs`

---

## Notes
This project is designed to demonstrate a complete ML system:
- reliable training pipeline
- experiment tracking
- production-style inference API
- user-facing frontend
- reproducible deployment

---
