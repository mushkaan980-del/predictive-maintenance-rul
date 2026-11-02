# Predictive Maintenance: RUL Prediction on NASA C-MAPSS

Predicts the remaining useful life (RUL) of turbofan engines from sensor data,
with uncertainty estimates, explanations, and a web dashboard.

## Status
Week 1: baseline model in progress

## Dataset
NASA C-MAPSS (FD001). Download it and place the files in `data/raw/`.

## Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt