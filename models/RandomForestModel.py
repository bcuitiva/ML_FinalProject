"""
PetWatch — Random Forest Model (Synthetic Dataset)
====================================================
Predicts abandonment risk level (HIGH / MEDIUM / LOW)
using the synthetic dataset with 15 variables.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

FEATURES = [
    'citizen_reports',
    'abuse_cases',
    'animals_seized',
    'vet_emergencies',
    'microchip_coverage',
    'sterilization_rate',
    'stray_density',
    'prior_cases_6m',
    'institutional_response_days',
]

FEATURE_LABELS = {
    'citizen_reports':             'Citizen Reports',
    'abuse_cases':                 'Abuse Cases',
    'animals_seized':              'Animals Seized',
    'vet_emergencies':             'Vet Emergencies',
    'microchip_coverage':          'Microchip Coverage (%)',
    'sterilization_rate':          'Sterilization Rate (%)',
    'stray_density':               'Stray Density',
    'prior_cases_6m':              'Prior Cases (6 months)',
    'institutional_response_days': 'Institutional Response (days)',
}

RISK_COLORS = {'HIGH': '#e85a6a', 'MEDIUM': '#e8935a', 'LOW': '#7de8a0'}
RISK_ORDER  = {'HIGH': 2, 'MEDIUM': 1, 'LOW': 0}
SYNTHETIC_CSV = 'petwatch_synthetic_dataset.csv'


def _resolve_csv(csv_path):
    if csv_path and os.path.exists(csv_path):
        return csv_path
    base = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(os.path.dirname(base), SYNTHETIC_CSV)
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(SYNTHETIC_CSV):
        return SYNTHETIC_CSV
    raise FileNotFoundError(f"Cannot find {SYNTHETIC_CSV}")


def get_results(csv_path=None):
    path = _resolve_csv(csv_path)
    df   = pd.read_csv(path).dropna(subset=FEATURES + ['abandonment_risk'])

    X = df[FEATURES].values
    y = df['abandonment_risk'].values

    model = RandomForestClassifier(
        n_estimators=200, random_state=42,
        max_depth=8, min_samples_leaf=5, n_jobs=-1
    )
    model.fit(X, y)

    preds   = model.predict(X)
    probas  = model.predict_proba(X)
    classes = list(model.classes_)

    # Global metrics
    acc  = round(float(accuracy_score(y, preds)) * 100, 1)
    prec = round(float(precision_score(y, preds, average='macro', zero_division=0)) * 100, 1)
    rec  = round(float(recall_score(y, preds, average='macro', zero_division=0)) * 100, 1)
    f1   = round(float(f1_score(y, preds, average='macro', zero_division=0)) * 100, 1)

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    cv_mean   = round(float(cv_scores.mean()) * 100, 1)
    cv_std    = round(float(cv_scores.std())  * 100, 1)

    # Confusion matrix
    cm_labels = ['HIGH', 'MEDIUM', 'LOW']
    cm        = confusion_matrix(y, preds, labels=cm_labels)

    # Per-class metrics
    per_class = {}
    for cls in cm_labels:
        mask = y == cls
        p  = round(float(precision_score(y, preds, labels=[cls], average='macro', zero_division=0)) * 100, 1)
        r  = round(float(recall_score(y, preds, labels=[cls], average='macro', zero_division=0)) * 100, 1)
        fi = round(float(f1_score(y, preds, labels=[cls], average='macro', zero_division=0)) * 100, 1)
        per_class[cls] = {'precision': p, 'recall': r, 'f1': fi, 'support': int(mask.sum())}

    # Feature importance — list of [name, pct] for JSON and Jinja
    importances_raw = sorted(
        [(FEATURE_LABELS[f], round(float(imp) * 100, 1))
         for f, imp in zip(FEATURES, model.feature_importances_)],
        key=lambda x: x[1], reverse=True
    )
    # Also as list of dicts for JS
    importances_dicts = [{'name': n, 'pct': p} for n, p in importances_raw]

    # Locality summary with all fields needed by random_forest.html
    df2 = df.copy()
    df2['pred_label'] = preds
    # Build proba lookup by index
    proba_by_idx = {}
    for i in range(len(preds)):
        proba_by_idx[i] = {c: round(float(probas[i][ci])*100, 1) for ci, c in enumerate(classes)}

    df2 = df2.reset_index(drop=True)
    locality_summary = []
    for loc, grp in df2.groupby('locality'):
        counts   = grp['pred_label'].value_counts()
        dominant = counts.index[0]
        grp_indices = grp.index.tolist()
        # Average proba for dominant class
        avg_proba_high = round(float(np.mean([proba_by_idx[i].get('HIGH', 0) for i in grp_indices])), 1)
        avg_stray      = round(float(grp['stray_density'].mean()), 1)
        avg_abuse      = round(float(grp['abuse_cases'].mean()), 1)
        avg_micro      = round(float(grp['microchip_coverage'].mean()), 1)
        avg_urgencias  = round(float(grp['vet_emergencies'].mean()), 1)
        locality_summary.append({
            'name':       loc,
            'label':      dominant,
            'color':      RISK_COLORS.get(dominant, '#fff'),
            'n':          len(grp),
            'pct_high':   round(100 * counts.get('HIGH',   0) / len(grp), 1),
            'pct_medium': round(100 * counts.get('MEDIUM', 0) / len(grp), 1),
            'pct_low':    round(100 * counts.get('LOW',    0) / len(grp), 1),
            # Legacy fields for random_forest.html table
            'risk_score':  avg_stray,
            'maltrato':    int(avg_abuse),
            'urgencias':   int(avg_urgencias),
            'microchip':   round(float(avg_micro), 1),
            'proba_high':  avg_proba_high,
        })
    locality_summary.sort(key=lambda x: RISK_ORDER.get(x['label'], 0), reverse=True)

    counts_total = {
        'HIGH':   int((preds == 'HIGH').sum()),
        'MEDIUM': int((preds == 'MEDIUM').sum()),
        'LOW':    int((preds == 'LOW').sum()),
    }

    return {
        # Table data
        'localities':          locality_summary,
        'importances':         importances_raw,       # list of (name, pct) tuples for Jinja {% for %}
        'importances_dicts':   importances_dicts,      # list of {name, pct} for JSON
        'counts':              counts_total,
        # Metrics
        'accuracy':            acc,
        'precision':           prec,
        'recall':              rec,
        'f1':                  f1,
        'cv_mean':             cv_mean,
        'cv_std':              cv_std,
        'confusion_matrix':    cm.tolist(),
        'cm_labels':           cm_labels,
        'per_class':           per_class,
        # Model info
        'n_estimators':        200,
        'max_depth':           8,
        'n_features':          len(FEATURES),
        'n_records':           len(df),
        'n_localities':        len(locality_summary),
    }
