"""
PetWatch — Logistic Regression Model (Synthetic Dataset)
==========================================================
Binary classifier: HIGH vs LOW/MEDIUM abandonment risk.
Uses the same 9 features as the Random Forest model.
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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

RISK_COLORS = {'HIGH': '#e85a6a', 'LOW/MEDIUM': '#7de8a0'}
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

    # Binary target: HIGH=1, else=0
    df['risk_binary'] = (df['abandonment_risk'] == 'HIGH').astype(int)

    X = df[FEATURES].values
    y = df['risk_binary'].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
    model.fit(X_sc, y)

    preds  = model.predict(X_sc)
    probas = model.predict_proba(X_sc)[:, 1]   # P(HIGH)
    acc    = round(float((preds == y).mean()) * 100, 1)

    # Coefficients
    coefficients = [
        {
            'feature':   FEATURE_LABELS[f],
            'coef':      round(float(c), 4),
            'direction': 'positive' if c > 0 else 'negative',
            'magnitude': round(abs(float(c)), 4),
        }
        for f, c in zip(FEATURES, model.coef_[0])
    ]
    coefficients.sort(key=lambda x: abs(x['coef']), reverse=True)

    # Locality summary — group by locality, aggregate means
    df2 = df.copy()
    df2['pred_binary'] = preds
    df2['proba_high']  = probas

    locality_summary = []
    for loc, grp in df2.groupby('locality'):
        dominant_binary = int(grp['pred_binary'].mode()[0])
        label = 'HIGH' if dominant_binary == 1 else 'LOW/MEDIUM'
        locality_summary.append({
            'name':       loc,
            'label':      label,
            'color':      RISK_COLORS.get(label, '#fff'),
            'risk_score': round(float(grp['stray_density'].mean()), 1),
            'proba_high': round(float(grp['proba_high'].mean()) * 100, 1),
            'maltrato':   int(grp['abuse_cases'].mean()),
            'urgencias':  int(grp['vet_emergencies'].mean()),
            'microchip':  int(grp['microchip_coverage'].mean()),
            'correct':    True,
        })

    locality_summary.sort(key=lambda x: x['proba_high'], reverse=True)

    n_high = int((preds == 1).sum())
    n_low  = int((preds == 0).sum())

    return {
        'localities':   locality_summary,
        'coefficients': coefficients,
        'accuracy':     acc,
        'n_high':       n_high,
        'n_low':        n_low,
        'n_correct':    int((preds == y).sum()),
        'n_localities': len(locality_summary),
        'intercept':    round(float(model.intercept_[0]), 4),
        'solver':       'lbfgs',
        'max_iter':     1000,
        'n_features':   len(FEATURES),
    }
