import os
import joblib
import pandas as pd
import numpy as np
import shap

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'credit_xgboost_model.pkl'))

_model = None
_explainer = None

def get_model():
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
        else:
            raise FileNotFoundError(f"Model XGBoost tidak ditemukan di: {MODEL_PATH}")
    return _model

def get_explainer():
    global _explainer
    if _explainer is None:
        model = get_model()
        _explainer = shap.TreeExplainer(model)
    return _explainer

def preprocess_applicant_input(raw_input: dict) -> pd.DataFrame:
    """
    Mengubah input mentah pemohon menjadi 22 fitur tabular yang siap diprediksi XGBoost.
    """
    age = float(raw_input.get('person_age', 25))
    income = float(raw_input.get('person_income', 50000))
    emp_len = float(raw_input.get('person_emp_length', 3.0))
    loan_amnt = float(raw_input.get('loan_amnt', 10000))
    int_rate = float(raw_input.get('loan_int_rate', 11.0))
    cred_hist = float(raw_input.get('cb_person_cred_hist_length', 4.0))
    home_ownership = str(raw_input.get('person_home_ownership', 'RENT')).upper()
    intent = str(raw_input.get('loan_intent', 'PERSONAL')).upper()
    grade = str(raw_input.get('loan_grade', 'B')).upper()
    default_on_file = str(raw_input.get('cb_person_default_on_file', 'N')).upper()

    # Derived Features
    loan_pct_income = loan_amnt / (income + 1e-5)
    total_cost = loan_amnt * (1.0 + int_rate / 100.0)
    disposable_inc = income - loan_amnt
    emp_to_age = emp_len / (age + 1e-5)
    cred_to_age = cred_hist / (age + 1e-5)
    
    grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    grade_enc = grade_map.get(grade, 1)
    default_enc = 1 if default_on_file == 'Y' else 0
    high_risk = 1 if (loan_pct_income > 0.35 or default_enc == 1 or grade_enc >= 4) else 0

    feature_dict = {
        'person_age': age,
        'person_income': income,
        'person_emp_length': emp_len,
        'loan_amnt': loan_amnt,
        'loan_int_rate': int_rate,
        'loan_percent_income': loan_pct_income,
        'cb_person_cred_hist_length': cred_hist,
        'total_loan_cost': total_cost,
        'disposable_income_est': disposable_inc,
        'emp_to_age_ratio': emp_to_age,
        'cred_hist_to_age_ratio': cred_to_age,
        'high_risk_flag': int(high_risk),
        'loan_grade_encoded': int(grade_enc),
        'cb_person_default_on_file_encoded': int(default_enc),
        'person_home_ownership_OTHER': 1 if home_ownership == 'OTHER' else 0,
        'person_home_ownership_OWN': 1 if home_ownership == 'OWN' else 0,
        'person_home_ownership_RENT': 1 if home_ownership == 'RENT' else 0,
        'loan_intent_EDUCATION': 1 if intent == 'EDUCATION' else 0,
        'loan_intent_HOMEIMPROVEMENT': 1 if intent == 'HOMEIMPROVEMENT' else 0,
        'loan_intent_MEDICAL': 1 if intent == 'MEDICAL' else 0,
        'loan_intent_PERSONAL': 1 if intent == 'PERSONAL' else 0,
        'loan_intent_VENTURE': 1 if intent == 'VENTURE' else 0,
    }
    return pd.DataFrame([feature_dict])

def predict_credit_risk(df_features: pd.DataFrame) -> dict:
    """
    Menghitung Probability of Default (PD), Risk Tier, dan SHAP values.
    """
    model = get_model()
    explainer = get_explainer()

    # 1. Hitung Probability of Default (PD)
    prob_default = float(model.predict_proba(df_features)[0][1])

    # 2. Risk Classification Tier
    if prob_default < 0.20:
        risk_level = 'LOW_RISK'
        recommendation = 'APPROVE'
        risk_color = '#10b981' # Green
    elif prob_default <= 0.40:
        risk_level = 'MEDIUM_RISK'
        recommendation = 'MANUAL_REVIEW'
        risk_color = '#f59e0b' # Amber
    else:
        risk_level = 'HIGH_RISK'
        recommendation = 'REJECT'
        risk_color = '#ef4444' # Red

    # 3. Hitung SHAP Values
    shap_explanation = explainer(df_features)
    shap_values = shap_explanation.values[0]
    feature_names = df_features.columns.tolist()

    # Format kontribusi fitur
    contributions = []
    for feat, val, shap_val in zip(feature_names, df_features.iloc[0], shap_values):
        contributions.append({
            'feature': feat,
            'value': val,
            'shap_value': float(shap_val),
            'abs_shap': abs(float(shap_val))
        })

    # Sort berdasarkan dampak terbesar
    sorted_contribs = sorted(contributions, key=lambda x: x['abs_shap'], reverse=True)
    
    # Pisahkan Top Drivers (+SHAP) dan Mitigating Factors (-SHAP)
    top_risk_drivers = [c for c in sorted_contribs if c['shap_value'] > 0][:3]
    mitigating_factors = [c for c in sorted_contribs if c['shap_value'] < 0][:3]

    return {
        'probability_of_default': prob_default,
        'risk_level': risk_level,
        'recommendation': recommendation,
        'risk_color': risk_color,
        'shap_values': shap_values,
        'shap_explanation': shap_explanation,
        'sorted_contributions': sorted_contribs,
        'top_risk_drivers': top_risk_drivers,
        'mitigating_factors': mitigating_factors,
        'features_df': df_features
    }
