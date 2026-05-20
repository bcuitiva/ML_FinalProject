"""
PetWatch — K-Means Clustering Model (Synthetic Dataset)
=========================================================
Unsupervised clustering of localities into 3 risk tiers.
Uses the same 9 features as the Random Forest model.
K = 3  (LOW / MEDIUM / HIGH risk clusters)
"""

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
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
    df   = pd.read_csv(path).dropna(subset=FEATURES)

    X      = df[FEATURES].values
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    model    = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = model.fit_predict(X_sc)
    df2      = df.copy()
    df2['cluster'] = clusters

    # Map clusters → risk labels by mean stray_density (higher = more risk)
    cluster_means = df2.groupby('cluster')['stray_density'].mean().sort_values()
    risk_map = {
        cluster_means.index[0]: 'LOW',
        cluster_means.index[1]: 'MEDIUM',
        cluster_means.index[2]: 'HIGH',
    }
    df2['risk_label'] = df2['cluster'].map(risk_map)

    # Cluster centroids
    centroids_sc = model.cluster_centers_
    centroids    = scaler.inverse_transform(centroids_sc)
    centroid_list = []
    for idx, risk in risk_map.items():
        c = centroids[idx]
        centroid_list.append({
            'label': risk,
            'color': RISK_COLORS[risk],
            **{FEATURE_LABELS[f]: round(float(v), 1) for f, v in zip(FEATURES, c)},
        })
    centroid_list.sort(key=lambda x: RISK_ORDER[x['label']], reverse=True)

    # Locality summary
    locality_summary = []
    for loc, grp in df2.groupby('locality'):
        label_counts = grp['risk_label'].value_counts()
        dominant = label_counts.index[0]
        locality_summary.append({
            'name':      loc,
            'label':     dominant,
            'color':     RISK_COLORS.get(dominant, '#fff'),
            'cluster':   int(grp['cluster'].mode()[0]),
            'maltrato':      int(grp['abuse_cases'].mean()),
            'urgencias':     int(grp['vet_emergencies'].mean()),
            'microchip':     round(float(grp['microchip_coverage'].mean()), 1),
            'stray_density': round(float(grp['stray_density'].mean()), 1),
            'pct_high':   round(100 * (grp['risk_label'] == 'HIGH').mean(), 1),
            'pct_medium': round(100 * (grp['risk_label'] == 'MEDIUM').mean(), 1),
            'pct_low':    round(100 * (grp['risk_label'] == 'LOW').mean(), 1),
        })

    locality_summary.sort(key=lambda x: RISK_ORDER.get(x['label'], 0), reverse=True)

    counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    for loc in locality_summary:
        counts[loc['label']] = counts.get(loc['label'], 0) + 1

    return {
        'localities':     locality_summary,
        'centroids':      centroid_list,
        'counts':         counts,
        'inertia':        round(float(model.inertia_), 2),
        'k':              3,
        'n_init':         10,
        'n_features':     len(FEATURES),
        'n_localities':   len(locality_summary),
        'feature_labels': list(FEATURE_LABELS.values()),
    }
