# Predictive Maintenance: RUL Prediction on NASA C-MAPSS

Predicts the remaining useful life (RUL) of turbofan engines from sensor data,
with uncertainty estimates, explanations, and a web dashboard.

## Status
Week 1: baseline model in progress

## Dataset
NASA C-MAPSS turbofan engine degradation simulation (FD001-FD004),
from the NASA Ames Prognostics Data Repository (Saxena & Goebel),
obtained via Kaggle. Place the files in `data/raw/`. Data is not included in this repo.

## Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt