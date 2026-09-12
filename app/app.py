import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils_ml import preprocess_applicant_input, predict_credit_risk
from utils_llm import generate_credit_memo

# ==============================================================================
# PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="XAI Credit Agent - Underwriting System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Banking Dashboard
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-approve { background-color: #d1fae5; color: #065f46; }
    .badge-review { background-color: #fef3c7; color: #92400e; }
    .badge-reject { background-color: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("<div class="main-header">🏦 XAI Credit Agent: Explainable AI & Credit Underwriting</div>", unsafe_allow_html=True)
st.markdown("<div class="sub-header">Sistem Penilaian Risiko Kredit & Pembuatan Memo Keputusan Otomatis berbasis <b>XGBoost (AUC: 0.95)</b>, <b>SHAP Interpretability</b>, dan <b>Fine-Tuned LLM (Qwen 2.5)</b></div>", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR: APPLICANT INPUT FORM & PRESETS
# ==============================================================================
st.sidebar.header("📋 Profil Pemohon Pinjaman")

# Preset Profiles for Instant Demo
preset_option = st.sidebar.selectbox(
    "💡 Pilih Contoh Profil Preset:",
    [
        "Custom Input (Manual)",
        "Contoh 1: Profil Risiko Rendah (Low Risk - Approving)",
        "Contoh 2: Profil Risiko Moderat (Medium Risk - Manual Review)",
        "Contoh 3: Profil Risiko Tinggi (High Risk - Reject)"
    ]
)

# Preset Values Configuration
if preset_option == "Contoh 1: Profil Risiko Rendah (Low Risk - Approving)":
    default_vals = {
        "age": 28, "income": 85000, "emp_len": 5.0, "home": "MORTGAGE",
        "intent": "VENTURE", "grade": "A", "amount": 8000, "rate": 7.5,
        "default_hist": "N", "cred_hist": 6
    }
elif preset_option == "Contoh 2: Profil Risiko Moderat (Medium Risk - Manual Review)":
    default_vals = {
        "age": 24, "income": 45000, "emp_len": 2.0, "home": "RENT",
        "intent": "MEDICAL", "grade": "C", "amount": 12000, "rate": 13.5,
        "default_hist": "N", "cred_hist": 3
    }
elif preset_option == "Contoh 3: Profil Risiko Tinggi (High Risk - Reject)":
    default_vals = {
        "age": 22, "income": 28000, "emp_len": 1.0, "home": "RENT",
        "intent": "DEBTCONSOLIDATION", "grade": "E", "amount": 16000, "rate": 18.5,
        "default_hist": "Y", "cred_hist": 2
    }
else:
    default_vals = {
        "age": 25, "income": 50000, "emp_len": 3.0, "home": "RENT",
        "intent": "PERSONAL", "grade": "B", "amount": 10000, "rate": 11.0,
        "default_hist": "N", "cred_hist": 4
    }

# Input Form
with st.sidebar.form("applicant_form"):
    st.subheader("Data Demografi & Finansial")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        age = st.number_input("Usia (Tahun)", min_value=18, max_value=85, value=default_vals["age"])
        emp_len = st.number_input("Masa Kerja (Thn)", min_value=0.0, max_value=50.0, value=float(default_vals["emp_len"]), step=0.5)
    with col_sb2:
        income = st.number_input("Pendapatan Tahunan ($)", min_value=1000, max_value=2000000, value=default_vals["income"], step=5000)
        cred_hist = st.number_input("Panjang Riwayat Kredit (Thn)", min_value=1, max_value=40, value=default_vals["cred_hist"])

    home_ownership = st.selectbox(
        "Status Kepemilikan Rumah",
        ["RENT", "MORTGAGE", "OWN", "OTHER"],
        index=["RENT", "MORTGAGE", "OWN", "OTHER"].index(default_vals["home"])
    )

    st.subheader("Detail Fasilitas Pinjaman")
    loan_intent = st.selectbox(
        "Tujuan Pinjaman",
        ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
        index=["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"].index(default_vals["intent"])
    )

    col_sb3, col_sb4 = st.columns(2)
    with col_sb3:
        loan_amnt = st.number_input("Besaran Pinjaman ($)", min_value=500, max_value=100000, value=default_vals["amount"], step=1000)
        loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"], index=["A", "B", "C", "D", "E", "F", "G"].index(default_vals["grade"]))
    with col_sb4:
        loan_int_rate = st.number_input("Suku Bunga (%)", min_value=3.0, max_value=30.0, value=float(default_vals["rate"]), step=0.25)
        default_hist = st.selectbox("Riwayat Gagal Bayar", ["N", "Y"], index=0 if default_vals["default_hist"] == "N" else 1)

    submit_btn = st.form_submit_button("⚡ Analisis Risiko Kredit", use_container_width=True)

# Settings in Sidebar
with st.sidebar.expander("⚙️ Pengaturan Engine"):
    llm_backend = st.selectbox("Engine Memo LLM:", ["auto", "local_adapter", "ollama"], index=0)
    ollama_endpoint = st.text_input("Ollama URL:", value="http://localhost:11434")

# ==============================================================================
# MAIN DASHBOARD PROCESSING
# ==============================================================================
raw_input = {
    "person_age": age,
    "person_income": income,
    "person_emp_length": emp_len,
    "person_home_ownership": home_ownership,
    "loan_intent": loan_intent,
    "loan_grade": loan_grade,
    "loan_amnt": loan_amnt,
    "loan_int_rate": loan_int_rate,
    "cb_person_default_on_file": default_hist,
    "cb_person_cred_hist_length": cred_hist
}

# Preprocessing & Predictive ML
df_features = preprocess_applicant_input(raw_input)
ml_result = predict_credit_risk(df_features)

pd_val = ml_result['probability_of_default']
risk_level = ml_result['risk_level']
recommendation = ml_result['recommendation']

# ------------------------------------------------------------------------------
# ROW 1: TOP KPI METRICS & RISK GAUGE
# ------------------------------------------------------------------------------
col_m1, col_m2, col_m3, col_m4 = st.columns([1.2, 1, 1, 1.3])

with col_m1:
    st.markdown("<div class="card">", unsafe_allow_html=True)
    st.markdown("<div class="metric-title">Rekomendasi Keputusan</div>", unsafe_allow_html=True)
    if recommendation == "APPROVE":
        st.markdown("<div class="metric-value" style="color: #059669;">✅ APPROVE</div>", unsafe_allow_html=True)
        st.markdown("<span class="badge badge-approve">Risiko Terkendali</span>", unsafe_allow_html=True)
    elif recommendation == "MANUAL_REVIEW":
        st.markdown("<div class="metric-value" style="color: #d97706;">⚠️ REVIEW</div>", unsafe_allow_html=True)
        st.markdown("<span class="badge badge-review">Verifikasi Tambahan</span>", unsafe_allow_html=True)
    else:
        st.markdown("<div class="metric-value" style="color: #dc2626;">❌ REJECT</div>", unsafe_allow_html=True)
        st.markdown("<span class="badge badge-reject">Risiko Tinggi</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_m2:
    st.markdown("<div class="card">", unsafe_allow_html=True)
    st.markdown("<div class="metric-title">Probabilitas Default (PD)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class="metric-value">{pd_val * 100:.1f}%</div>", unsafe_allow_html=True)
    st.caption("Prediksi XGBoost Model (AUC: 0.95)")
    st.markdown("</div>", unsafe_allow_html=True)

with col_m3:
    loan_pct_income = (loan_amnt / income) * 100.0
    st.markdown("<div class="card">", unsafe_allow_html=True)
    st.markdown("<div class="metric-title">Rasio Pinjaman/Gaji</div>", unsafe_allow_html=True)
    st.markdown(f"<div class="metric-value">{loan_pct_income:.1f}%</div>", unsafe_allow_html=True)
    if loan_pct_income > 35:
        st.caption("⚠️ Melebihi ambang batas 35%")
    else:
        st.caption("✅ Dalam batas aman")
    st.markdown("</div>", unsafe_allow_html=True)

with col_m4:
    # Gauge Chart for Probability of Default
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pd_val * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': ml_result['risk_color']},
            'steps': [
                {'range': [0, 20], 'color': "#dcfce7"},
                {'range': [20, 40], 'color': "#fef3c7"},
                {'range': [40, 100], 'color': "#fee2e2"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 0.75,
                'value': pd_val * 100
            }
        }
    ))
    fig_gauge.update_layout(margin=dict(l=10, r=10, t=25, b=10), height=140)
    st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")

# ------------------------------------------------------------------------------
# ROW 2: EXPLAINABLE AI (SHAP WATERFALL & FEATURE ATTRIBUTION)
# ------------------------------------------------------------------------------
col_left, col_right = st.columns([1.1, 1.3])

with col_left:
    st.subheader("🔍 Explainable AI: Analisis Kontribusi SHAP")
    st.write("Transparansi matematis kontribusi setiap variabel terhadap peningkatan (+SHAP) atau peredam (-SHAP) risiko gagal bayar:")

    # Prepare DataFrame for SHAP horizontal bar chart
    top_contribs = ml_result['sorted_contributions'][:8]
    feat_df = pd.DataFrame(top_contribs)
    feat_df['type'] = feat_df['shap_value'].apply(lambda x: 'Pendorong Risiko (+SHAP)' if x > 0 else 'Pereda Risiko (-SHAP)')
    feat_df = feat_df.sort_values(by='shap_value', ascending=True)

    fig_shap = px.bar(
        feat_df,
        x='shap_value',
        y='feature',
        orientation='h',
        color='type',
        color_discrete_map={
            'Pendorong Risiko (+SHAP)': '#ef4444',
            'Pereda Risiko (-SHAP)': '#10b981'
        },
        labels={'shap_value': 'SHAP Contribution Value', 'feature': 'Variabel Fitur'},
        title="Top 8 Fitur Berpengaruh (TreeExplainer)"
    )
    fig_shap.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=350, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_shap, use_container_width=True)

with col_right:
    st.subheader("📄 Automated Credit Underwriting Memo (Fine-Tuned LLM)")
    
    # Generate Memo with Qwen 2.5 / Rule-Grounded Fallback
    with st.spinner("Menyusun memo pertimbangan kredit dengan AI Underwriter..."):
        memo_json = generate_credit_memo(raw_input, ml_result, mode=llm_backend, ollama_url=ollama_endpoint)

    st.markdown(f"**Engine Sumber:** `{memo_json.get('source', 'XAI LLM Engine')}`")

    # Executive Summary Card
    st.markdown("<div class="card">", unsafe_allow_html=True)
    st.markdown("##### 📌 Executive Summary")
    st.write(memo_json.get('executive_summary', '-'))
    st.markdown("</div>", unsafe_allow_html=True)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("<div class="card">", unsafe_allow_html=True)
        st.markdown("##### ⚠️ Faktor Pendorong Risiko (+SHAP)")
        for item in memo_json.get('key_risk_drivers', []):
            st.markdown(f"- {item}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_d2:
        st.markdown("<div class="card">", unsafe_allow_html=True)
        st.markdown("##### 🛡️ Faktor Mitigasi / Pereda Risiko (-SHAP)")
        for item in memo_json.get('mitigating_factors', []):
            st.markdown(f"- {item}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Mitigation Plan / Conditions
    if 'conditions_or_mitigation_plan' in memo_json and memo_json['conditions_or_mitigation_plan']:
        st.markdown("<div class="card">", unsafe_allow_html=True)
        st.markdown("##### 📋 Rekomendasi Syarat Pencairan / Mitigasi")
        for cond in memo_json['conditions_or_mitigation_plan']:
            st.markdown(f"1. {cond}")
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# ROW 3: RAW JSON & AUDIT TRAIL EXPORT
# ------------------------------------------------------------------------------
with st.expander("🛠️ Audit Trail & Raw JSON Output"):
    st.json(memo_json)
    st.download_button(
        label="📥 Download Credit Memo (JSON)",
        data=json.dumps(memo_json, indent=2),
        file_name=f"credit_memo_{int(income)}_{int(loan_amnt)}.json",
        mime="application/json"
    )
