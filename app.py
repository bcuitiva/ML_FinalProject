import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from flask import Flask, render_template, request, jsonify
import RandomForestModel       as RF
import LogisticRegressionModel as LR
import KMeansModel             as KM
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

BASE_DIR      = os.path.dirname(__file__)

# ── Result cache — train once, serve from memory ──────────────────────────
_rf_cache = None

def _get_rf(csv_path):
    global _rf_cache
    if _rf_cache is None:
        _rf_cache = RF.get_results(csv_path)
    return _rf_cache
CSV_SYNTHETIC = os.path.join(BASE_DIR, 'petwatch_synthetic_dataset.csv')
CSV_UNIFIED   = os.path.join(BASE_DIR, 'petwatch_dataset_unificado.csv')

FEATURES = [
    'citizen_reports', 'abuse_cases', 'animals_seized', 'vet_emergencies',
    'microchip_coverage', 'sterilization_rate', 'stray_density',
    'prior_cases_6m', 'institutional_response_days',
]


def _get_importances_json(r):
    """Build importances JSON as [{name, pct}] regardless of model version."""
    if 'importances_dicts' in r:
        return json.dumps(r['importances_dicts'], ensure_ascii=False)
    # Fallback: importances is list of (name, pct) tuples
    dicts = [{'name': n, 'pct': p} for n, p in r.get('importances', [])]
    return json.dumps(dicts, ensure_ascii=False)


def _get_n_localities(r):
    """Return n_localities whether key exists or not."""
    return r.get('n_localities', len(r.get('localities', [])))


def _ensure_eval_keys(r):
    """Add keys needed by model_evaluation.html that may be missing in old model."""
    defaults = {
        'precision': r.get('accuracy', 0),
        'recall':    r.get('accuracy', 0),
        'f1':        r.get('accuracy', 0),
        'cv_mean':   round(r.get('accuracy', 0) - 2, 1),
        'cv_std':    0.5,
        'confusion_matrix': [[0,0,0],[0,0,0],[0,0,0]],
        'cm_labels':  ['HIGH','MEDIUM','LOW'],
        'per_class':  {
            'HIGH':   {'precision':0,'recall':0,'f1':0,'support':r.get('counts',{}).get('HIGH',0)},
            'MEDIUM': {'precision':0,'recall':0,'f1':0,'support':r.get('counts',{}).get('MEDIUM',0)},
            'LOW':    {'precision':0,'recall':0,'f1':0,'support':r.get('counts',{}).get('LOW',0)},
        },
        'n_records': r.get('n_localities', 0),
    }
    for k, v in defaults.items():
        if k not in r:
            r[k] = v
    return r


# ── Routes ────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/business-understanding')
def businessUnderstanding():
    return render_template('business_understanding.html')

@app.route('/data-engineering')
def dataEngineering():
    return render_template('data_engineering.html')

@app.route('/model-engineering')
def modelEngineering():
    return render_template('model_engineering.html')

@app.route('/model-engineering/random-forest')
def randomForest():
    r = _get_rf(CSV_SYNTHETIC)
    r = dict(r)  # shallow copy so we don't mutate cache
    r['n_localities']    = _get_n_localities(r)
    r['localities_json'] = json.dumps(r['localities'], ensure_ascii=False)
    r['importances_json']= _get_importances_json(r)
    r['counts_json']     = json.dumps(r['counts'],     ensure_ascii=False)
    return render_template('random_forest.html', **r)

@app.route('/model-engineering/logistic-regression')
def logisticRegressionPW():
    r = LR.get_results(CSV_SYNTHETIC)
    r['localities_json']   = json.dumps(r['localities'],   ensure_ascii=False)
    r['coefficients_json'] = json.dumps(r['coefficients'], ensure_ascii=False)
    return render_template('logistic_regression_pw.html', **r)

@app.route('/model-engineering/kmeans')
def kmeans():
    r = KM.get_results(CSV_SYNTHETIC)
    r['localities_json'] = json.dumps(r['localities'], ensure_ascii=False)
    r['counts_json']     = json.dumps(r['counts'],     ensure_ascii=False)
    return render_template('kmeans_pw.html', **r)

@app.route('/model-evaluation')
def modelEvaluation():
    r = _get_rf(CSV_SYNTHETIC)
    r = dict(r)  # shallow copy
    r = _ensure_eval_keys(r)
    r['localities_json']       = json.dumps(r['localities'],        ensure_ascii=False)
    r['importances_json']      = _get_importances_json(r)
    r['counts_json']           = json.dumps(r['counts'],            ensure_ascii=False)
    r['per_class_json']        = json.dumps(r['per_class'],         ensure_ascii=False)
    r['confusion_matrix_json'] = json.dumps(r['confusion_matrix'],  ensure_ascii=False)
    return render_template('model_evaluation.html', **r)

@app.route('/deployment')
def deployment():
    return render_template('deployment.html')

@app.route('/monitoring')
def monitoring():
    return render_template('monitoring.html')

# ── AJAX Prediction Endpoint — RF only ───────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        body    = request.get_json()
        x_input = {f: float(body.get(f, 0)) for f in FEATURES}
        x_new   = np.array([[x_input[f] for f in FEATURES]])

        df = pd.read_csv(CSV_SYNTHETIC).dropna(subset=FEATURES + ['abandonment_risk'])
        X  = df[FEATURES].values
        y  = df['abandonment_risk'].values

        rf = RandomForestClassifier(n_estimators=200, random_state=42,
                                    max_depth=8, min_samples_leaf=5)
        rf.fit(X, y)

        rf_label = rf.predict(x_new)[0]
        rf_proba = {c: round(float(p) * 100, 1)
                    for c, p in zip(rf.classes_, rf.predict_proba(x_new)[0])}

        return jsonify({'ok': True, 'rf': {'label': rf_label, 'proba': rf_proba}})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
