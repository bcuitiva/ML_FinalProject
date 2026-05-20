# PetWatch

**Machine Learning system for predicting animal abandonment risk across Bogotá localities.**

PetWatch applies the CRISP-ML(Q) methodology across 6 phases to identify which urban zones face the highest risk of animal abandonment, enabling data-driven resource allocation for animal welfare institutions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [CRISP-ML(Q) Phases](#crisp-mlq-phases)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Machine Learning Models](#machine-learning-models)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Team](#team)

---

## Project Overview

PetWatch is a full-stack web application developed as an academic project for the **Machine Learning** course at Universidad de Cundinamarca (6th semester, Software Engineering). It combines real open datasets from Bogotá and Cali with a purpose-built synthetic dataset to train a Random Forest classifier that predicts abandonment risk levels — **HIGH**, **MEDIUM**, or **LOW** — per urban locality.

The system is presented as an interactive web platform where each tab corresponds to a phase of the CRISP-ML(Q) lifecycle, allowing users to explore the data pipeline, run live predictions, and review model evaluation results.

---

## CRISP-ML(Q) Phases

| Phase | Page | Description |
|-------|------|-------------|
| P1 | Business Understanding | Problem definition, objectives, success metrics, and project scope |
| P2 | Data Engineering | Source datasets, unification pipeline, schema, and synthetic dataset justification |
| P3 | Model Engineering | Live prediction form + detailed pages for all 3 models |
| P4 | Model Evaluation | Metrics, confusion matrix, cross-validation, feature importance, model comparison |
| P5 | Deployment | System architecture, tech stack, REST API docs, Render.com deployment guide |
| P6 | Monitoring | Accuracy trend simulation, drift detection, retraining protocol, alert thresholds |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, Flask |
| Machine Learning | scikit-learn (RandomForest, LogisticRegression, KMeans) |
| Data Processing | pandas, numpy |
| Templating | Jinja2 |
| Frontend | Vanilla CSS, Vanilla JavaScript, Bootstrap 5, Chart.js |
| Fonts | Outfit, DM Sans, JetBrains Mono (Google Fonts) |
| Hosting | Render.com (gunicorn) |

---

## Project Structure

```
Petwatch/
├── app.py                          # Flask application — routes and RF result cache
├── requirements.txt                # Python dependencies
├── petwatch_synthetic_dataset.csv  # Synthetic dataset — used for all ML training
├── petwatch_dataset_unificado.csv  # Unified real dataset — used for Phase 2 documentation
├── models/
│   ├── RandomForestModel.py        # RF classifier — PRIMARY model
│   ├── LogisticRegressionModel.py  # LR binary classifier — baseline
│   └── KMeansModel.py              # KMeans clustering — unsupervised tier grouping
└── templates/
    ├── home.html                   # Landing page
    ├── business_understanding.html # Phase 1
    ├── data_engineering.html       # Phase 2
    ├── model_engineering.html      # Phase 3 — prediction form
    ├── random_forest.html          # Phase 3 — RF detail
    ├── logistic_regression_pw.html # Phase 3 — LR detail
    ├── kmeans_pw.html              # Phase 3 — KMeans detail
    ├── model_evaluation.html       # Phase 4
    ├── deployment.html             # Phase 5
    └── monitoring.html             # Phase 6
```

---

## Datasets

### Unified Real Dataset (`petwatch_dataset_unificado.csv`)

Compiled from **5 open institutional sources** covering Bogotá and Cali:

| Source | Description |
|--------|-------------|
| `CALI_ADOPTADOS` | Animal adoption records from Cali |
| `BOGOTA_BRIGADA` | Sterilization and microchip brigade records |
| `BOGOTA_MALTRATO` | Animal abuse cases attended by authorities |
| `BOGOTA_URGENCIA` | Emergency veterinary interventions |
| `BOGOTA_MICROCHIP` | Microchip registration records |

- **Records:** 17,742
- **Columns:** 16 (`fuente`, `ciudad`, `fecha`, `especie`, `localidad_zona`, `barrio`, `evento`, `atenciones`, `caninos`, `felinos`, `maltrato_atendidos`, `maltrato_aprehendidos`, `urgencias`, `microchip`, `castrados`, `potencialmente_peligroso`)
- **Cities:** Bogotá, Cali
- **Purpose:** Exploratory analysis and feature selection only — not used for model training due to high noise, missing values, and weak label correlations

### Synthetic Dataset (`petwatch_synthetic_dataset.csv`)

Purpose-built dataset designed to overcome the limitations of the unified real data for supervised ML training.

- **Records:** 6,500
- **Columns:** 15
- **Localities:** 20 Bogotá localities
- **Years:** 2021–2024
- **Target variable:** `abandonment_risk` (HIGH / MEDIUM / LOW)

| Variable | Description |
|----------|-------------|
| `locality` | Bogotá locality name |
| `species` | Animal species (dog, cat, bird, rabbit, other) |
| `month` | Month of record (1–12) |
| `year` | Year of record (2021–2024) |
| `socioeconomic_stratum` | Socioeconomic level (1–6) |
| `citizen_reports` | Number of citizen-filed stray animal reports |
| `abuse_cases` | Confirmed animal abuse incidents |
| `animals_seized` | Animals apprehended by authorities |
| `vet_emergencies` | Emergency veterinary interventions |
| `microchip_coverage` | Percentage of microchipped animals — strongest predictor (44%) |
| `sterilization_rate` | Percentage of sterilized animals in the zone |
| `stray_density` | Estimated stray animals per km² — 2nd predictor (29%) |
| `prior_cases_6m` | Cases reported in the last 6 months |
| `institutional_response_days` | Average days to institutional response |
| `abandonment_risk` | **TARGET** — HIGH / MEDIUM / LOW |

**Class distribution:** HIGH: 2,633 — MEDIUM: 3,037 — LOW: 830

**Why synthetic outperforms unified for training:** the unified dataset contains real-world noise, duplicates, and weak feature-to-label correlations. By engineering explicit causal relationships (e.g. `microchip_coverage` negatively correlated with risk, `stray_density` positively correlated), the Random Forest achieves **96.8% training accuracy** and **94.3% cross-validated accuracy**, compared to ~72% on the unified dataset.

---

## Machine Learning Models

All three models are trained on the **synthetic dataset** using the same 9 input features:

```
citizen_reports, abuse_cases, animals_seized, vet_emergencies,
microchip_coverage, sterilization_rate, stray_density,
prior_cases_6m, institutional_response_days
```

### Random Forest — PRIMARY MODEL

Multi-class classifier predicting HIGH / MEDIUM / LOW risk per record.

| Metric | Value |
|--------|-------|
| Training Accuracy | 96.8% |
| CV Accuracy (5-fold) | 94.3% ± 0.7% |
| Macro Precision | 96.9% |
| Macro Recall | 96.6% |
| Macro F1-Score | 96.8% |
| Estimators | 200 trees |
| Max Depth | 8 |

**Top feature importances:**

| Feature | Importance |
|---------|-----------|
| Microchip Coverage | 44.0% |
| Stray Density | 29.4% |
| Vet Emergencies | 9.1% |
| Abuse Cases | 8.8% |
| Citizen Reports | 4.3% |

### Logistic Regression — BASELINE

Binary classifier: HIGH risk vs LOW/MEDIUM risk.

- Solver: lbfgs, max_iter: 1000
- Features standardized with `StandardScaler`
- Training Accuracy: ~97% (binary task)
- Interpretable via signed coefficients

### K-Means Clustering — UNSUPERVISED

Groups localities into 3 risk tiers without predefined labels. Clusters are mapped to HIGH / MEDIUM / LOW based on mean `stray_density` of each cluster.

- K = 3, n_init = 10, random_state = 42
- Labels emerge from data structure — not predefined
- Used for geographic risk pattern discovery

---

## Installation & Setup

**Requirements:** Python 3.9+

```bash
# 1. Clone the repository
git clone https://github.com/your-username/petwatch.git
cd petwatch/Petwatch

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`**
```
flask
pandas
scikit-learn
numpy
```

---

## Running the Application

```bash
# From the Petwatch/ directory (where app.py lives)
python app.py
```

Open your browser at: **http://localhost:5000**

> **Note:** The first page load that triggers model training (Phase 3 or Phase 4) will take a few seconds while the Random Forest trains on 6,500 records with 200 trees and 5-fold cross-validation. Subsequent visits are served from an in-memory cache with no retraining.

---

## API Reference

### `POST /api/predict`

Predicts abandonment risk level for a given set of zone welfare indicators using the Random Forest model.

**Request body (JSON):**

```json
{
  "citizen_reports": 45,
  "abuse_cases": 12,
  "animals_seized": 8,
  "vet_emergencies": 20,
  "microchip_coverage": 0.25,
  "sterilization_rate": 0.40,
  "stray_density": 85.5,
  "prior_cases_6m": 30,
  "institutional_response_days": 7
}
```

**Response (JSON):**

```json
{
  "ok": true,
  "rf": {
    "label": "HIGH",
    "proba": {
      "HIGH": 82.4,
      "MEDIUM": 14.1,
      "LOW": 3.5
    }
  }
}
```

**Error response:**

```json
{
  "ok": false,
  "error": "error description"
}
```

---

## Deployment

PetWatch is deployed on **Render.com** as a Web Service.

| Setting | Value |
|---------|-------|
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `gunicorn app:app` |
| Auto-Deploy | On every push to `main` |

The synthetic CSV is bundled in the repository. The RF model retrains in-memory on cold start — no serialized model file is required.

> **Free tier note:** Render's free tier spins down after inactivity. The first request after a cold start may take 30–60 seconds.

---

## Team

| Name | Role |
|------|------|
| Elkin Yamith Almonacid López | Developer |
| Brayan David Cuitiva Umbarila | Developer |
| Brayan Yair Mendez Rodriguez | Developer |

**Course:** Machine Learning — 6th Semester, Systems and computation engineering  
**Institution:** Universidad de Cundinamarca  
**Methodology:** CRISP-ML(Q)
