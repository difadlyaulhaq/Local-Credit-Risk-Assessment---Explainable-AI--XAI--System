"""
XAI Credit Agent - Explainable AI Credit Underwriting System
================================================================
Streamlit dashboard for credit risk scoring (XGBoost), SHAP-based
explainability, and automated credit memo generation.
"""

import json
from dataclasses import dataclass

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils_ml import preprocess_applicant_input, predict_credit_risk
from utils_llm import generate_credit_memo, check_ollama_availability, discover_available_engines

# ==============================================================================
# CONSTANTS & TRANSLATIONS
# ==============================================================================
HOME_OPTIONS = ["RENT", "MORTGAGE", "OWN", "OTHER"]
INTENT_OPTIONS = ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
GRADE_OPTIONS = ["A", "B", "C", "D", "E", "F", "G"]

HOME_LABELS = {
    "RENT": "Sewa (Rent)",
    "MORTGAGE": "Kredit Rumah (Mortgage)",
    "OWN": "Milik Sendiri (Own)",
    "OTHER": "Lainnya (Other)",
}

INTENT_LABELS = {
    "PERSONAL": "Kebutuhan Pribadi",
    "EDUCATION": "Pendidikan",
    "MEDICAL": "Kesehatan / Medis",
    "VENTURE": "Modal Usaha / Bisnis",
    "HOMEIMPROVEMENT": "Renovasi Rumah",
    "DEBTCONSOLIDATION": "Konsolidasi Utang",
}

FEATURE_NAME_MAP = {
    "person_age": "Usia Pemohon",
    "person_income": "Pendapatan Tahunan",
    "person_emp_length": "Masa Kerja (Thn)",
    "loan_amnt": "Jumlah Pinjaman",
    "loan_int_rate": "Suku Bunga (%)",
    "loan_percent_income": "Rasio Pinjaman/Gaji",
    "cb_person_cred_hist_length": "Riwayat Kredit (Thn)",
    "total_loan_cost": "Total Beban Pinjaman",
    "disposable_income_est": "Estimasi Sisa Pendapatan",
    "emp_to_age_ratio": "Rasio Kerja/Usia",
    "cred_hist_to_age_ratio": "Rasio Kredit/Usia",
    "high_risk_flag": "Flag Risiko Tinggi",
    "loan_grade_encoded": "Peringkat Pinjaman (Grade)",
    "cb_person_default_on_file_encoded": "Riwayat Pernah Gagal Bayar",
    "person_home_ownership_OTHER": "Rumah: Lainnya",
    "person_home_ownership_OWN": "Rumah: Milik Sendiri",
    "person_home_ownership_RENT": "Rumah: Sewa",
    "loan_intent_EDUCATION": "Tujuan: Pendidikan",
    "loan_intent_HOMEIMPROVEMENT": "Tujuan: Renovasi",
    "loan_intent_MEDICAL": "Tujuan: Medis",
    "loan_intent_PERSONAL": "Tujuan: Pribadi",
    "loan_intent_VENTURE": "Tujuan: Bisnis/Usaha",
}

# Standar suku bunga rata-rata pasar per grade kredit (sebagai referensi komparasi)
MARKET_GRADE_BENCHMARKS = {
    "A": 7.25,
    "B": 10.75,
    "C": 13.50,
    "D": 15.75,
    "E": 18.25,
    "F": 20.50,
    "G": 22.75,
}

PRESETS = {
    "Custom Input (Manual)": {
        "age": 25, "income": 50000, "emp_len": 3.0, "home": "RENT",
        "intent": "PERSONAL", "grade": "B", "amount": 10000, "rate": 11.0,
        "default_hist": "N", "cred_hist": 4,
    },
    "Contoh 1: Profil Risiko Rendah (Low Risk - Approving)": {
        "age": 28, "income": 85000, "emp_len": 5.0, "home": "MORTGAGE",
        "intent": "VENTURE", "grade": "A", "amount": 8000, "rate": 7.5,
        "default_hist": "N", "cred_hist": 6,
    },
    "Contoh 2: Profil Risiko Moderat (Medium Risk - Manual Review)": {
        "age": 24, "income": 45000, "emp_len": 2.0, "home": "RENT",
        "intent": "MEDICAL", "grade": "C", "amount": 12000, "rate": 13.5,
        "default_hist": "N", "cred_hist": 3,
    },
    "Contoh 3: Profil Risiko Tinggi (High Risk - Reject)": {
        "age": 22, "income": 28000, "emp_len": 1.0, "home": "RENT",
        "intent": "DEBTCONSOLIDATION", "grade": "E", "amount": 16000, "rate": 18.5,
        "default_hist": "Y", "cred_hist": 2,
    },
}

RECOMMENDATION_STYLES = {
    "APPROVE": {
        "label": "DISETUJUI (APPROVE)",
        "badge_icon": "✅",
        "color": "#059669",
        "bg_color": "#ecfdf5",
        "border_color": "#a7f3d0",
        "badge_class": "badge-approve",
        "badge_text": "Risiko Terkendali",
        "desc": "Aplikasi memenuhi standar mitigasi risiko kredit. Probabilitas gagal bayar berada pada zona aman.",
    },
    "MANUAL_REVIEW": {
        "label": "PERLU REVIEW MANUAL",
        "badge_icon": "⚠️",
        "color": "#d97706",
        "bg_color": "#fffbeb",
        "border_color": "#fde68a",
        "badge_class": "badge-review",
        "badge_text": "Verifikasi Tambahan",
        "desc": "Profil berada di zona perantara. Disarankan verifikasi dokumen tambahan atau penyesuaian tenor.",
    },
    "REJECT": {
        "label": "DITOLAK (REJECT)",
        "badge_icon": "❌",
        "color": "#dc2626",
        "bg_color": "#fef2f2",
        "border_color": "#fecaca",
        "badge_class": "badge-reject",
        "badge_text": "Risiko Terlalu Tinggi",
        "desc": "Indikator risiko kredit melebihi batas toleransi. Probabilitas default tergolong tinggi.",
    },
}

LOAN_TO_INCOME_WARNING_THRESHOLD = 35.0


@dataclass
class ApplicantInput:
    """Typed container for a single loan application."""
    age: int
    income: int
    emp_len: float
    home_ownership: str
    intent: str
    grade: str
    amount: int
    rate: float
    default_hist: str
    cred_hist: int

    def to_raw_dict(self) -> dict:
        return {
            "person_age": self.age,
            "person_income": self.income,
            "person_emp_length": self.emp_len,
            "person_home_ownership": self.home_ownership,
            "loan_intent": self.intent,
            "loan_grade": self.grade,
            "loan_amnt": self.amount,
            "loan_int_rate": self.rate,
            "cb_person_default_on_file": self.default_hist,
            "cb_person_cred_hist_length": self.cred_hist,
        }


# ==============================================================================
# PAGE CONFIGURATION & LIGHT SKY BLUE STYLING
# ==============================================================================
def configure_page() -> None:
    st.set_page_config(
        page_title="XAI Credit Agent - Underwriting System",
        page_icon="🌤️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

            html, body, [class*="css"], .stMarkdown {
                font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }

            .stApp {
                background-color: #f8fafc;
            }

            /* ---------- Header Bar with Light Sky Blue Gradient ---------- */
            .header-container {
                background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #bae6fd 100%);
                border: 1px solid #bae6fd;
                border-radius: 16px;
                padding: 1.4rem 1.8rem;
                margin-bottom: 1.2rem;
                box-shadow: 0 4px 15px rgba(56, 189, 248, 0.08);
            }
            .main-header {
                font-size: 2.1rem;
                font-weight: 800;
                background: linear-gradient(90deg, #0284c7 0%, #0369a1 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 0;
                letter-spacing: -0.02em;
            }
            .sub-header {
                font-size: 0.95rem;
                color: #475569;
                margin-top: 0.35rem;
                line-height: 1.5;
            }
            .tech-pill {
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid #7dd3fc;
                color: #0369a1;
                font-size: 0.76rem;
                font-weight: 600;
                padding: 0.2rem 0.65rem;
                border-radius: 9999px;
                margin-top: 0.5rem;
                margin-right: 0.35rem;
            }

            /* ---------- Cards & Containers ---------- */
            .card {
                background-color: #ffffff;
                border-radius: 14px;
                padding: 1.15rem 1.3rem;
                border: 1px solid #e2e8f0;
                box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
                margin-bottom: 0.85rem;
                transition: transform 0.15s ease, box-shadow 0.15s ease;
            }
            .card:hover {
                box-shadow: 0 6px 16px rgba(14, 165, 233, 0.08);
                border-color: #bae6fd;
            }

            /* ---------- Metric Card Highlights ---------- */
            .metric-card {
                background: #ffffff;
                border-radius: 14px;
                padding: 1.15rem;
                border: 1px solid #e2e8f0;
                box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                min-height: 135px;
            }
            .metric-card-sky {
                background: linear-gradient(180deg, #ffffff 0%, #f0f9ff 100%);
                border-color: #bae6fd;
            }
            .metric-title {
                font-size: 0.78rem;
                font-weight: 600;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 0.3rem;
                display: flex;
                align-items: center;
                gap: 0.4rem;
            }
            .metric-value {
                font-size: 1.8rem;
                font-weight: 800;
                line-height: 1.15;
                color: #0f172a;
            }
            .metric-subtitle {
                font-size: 0.78rem;
                color: #64748b;
                margin-top: 0.35rem;
                font-weight: 500;
            }

            /* ---------- Decision Hero Banner ---------- */
            .decision-hero {
                border-radius: 14px;
                padding: 1.2rem 1.4rem;
                border-width: 1.5px;
                border-style: solid;
                margin-bottom: 0.85rem;
                display: flex;
                flex-direction: column;
                justify-content: center;
                min-height: 135px;
            }
            .decision-header {
                font-size: 0.78rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                margin-bottom: 0.2rem;
            }
            .decision-title {
                font-size: 1.45rem;
                font-weight: 800;
                letter-spacing: -0.01em;
                margin-bottom: 0.2rem;
            }
            .decision-desc {
                font-size: 0.83rem;
                margin-top: 0.2rem;
                line-height: 1.4;
            }

            /* ---------- Badges ---------- */
            .badge {
                display: inline-flex;
                align-items: center;
                gap: 0.3rem;
                padding: 0.22rem 0.7rem;
                border-radius: 9999px;
                font-weight: 700;
                font-size: 0.74rem;
                letter-spacing: 0.02em;
            }
            .badge-approve { background-color: #d1fae5; color: #065f46; border: 1px solid #a7f3d0; }
            .badge-review  { background-color: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
            .badge-reject  { background-color: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
            .badge-sky     { background-color: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd; }

            /* ---------- Section Titles ---------- */
            .section-title {
                font-size: 1.12rem;
                font-weight: 700;
                color: #0f172a;
                margin: 0.35rem 0 0.65rem 0;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }

            /* ---------- Progress Bar for PD ---------- */
            .progress-container {
                width: 100%;
                background-color: #e2e8f0;
                border-radius: 9999px;
                height: 8px;
                margin-top: 0.45rem;
                overflow: hidden;
            }
            .progress-bar {
                height: 100%;
                border-radius: 9999px;
                transition: width 0.3s ease;
            }

            /* ================================================================= */
            /* SIDEBAR: HIGH-CONTRAST SOLID BLACK TEXT STYLING                   */
            /* ================================================================= */
            section[data-testid="stSidebar"] {
                background-color: #ffffff !important;
                border-right: 1px solid #e2e8f0 !important;
            }
            section[data-testid="stSidebar"] * {
                color: #0f172a !important; /* Force solid black/dark text across all elements */
            }
            section[data-testid="stSidebar"] h1,
            section[data-testid="stSidebar"] h2,
            section[data-testid="stSidebar"] h3,
            section[data-testid="stSidebar"] h4,
            section[data-testid="stSidebar"] h5,
            section[data-testid="stSidebar"] h6 {
                color: #000000 !important;
                font-weight: 800 !important;
            }
            section[data-testid="stSidebar"] label,
            section[data-testid="stSidebar"] label p,
            section[data-testid="stSidebar"] .stWidgetLabel p,
            section[data-testid="stSidebar"] .stMarkdown p {
                color: #0f172a !important;
                font-weight: 700 !important;
                font-size: 0.88rem !important;
            }
            section[data-testid="stSidebar"] .stMarkdown strong {
                color: #000000 !important;
                font-weight: 800 !important;
            }
            section[data-testid="stSidebar"] input {
                color: #0f172a !important;
                font-weight: 600 !important;
                background-color: #ffffff !important;
                border: 1px solid #cbd5e1 !important;
            }
            section[data-testid="stSidebar"] div[data-baseweb="input"] {
                background-color: #ffffff !important;
                background: #ffffff !important;
                border-color: #cbd5e1 !important;
            }
            section[data-testid="stSidebar"] div[data-baseweb="select"] div {
                color: #0f172a !important;
                font-weight: 600 !important;
                background-color: #ffffff !important;
            }
            section[data-testid="stSidebar"] div[data-baseweb="select"] span {
                color: #0f172a !important;
                font-weight: 600 !important;
            }
            section[data-testid="stSidebar"] div[data-testid="stExpander"] details summary span {
                color: #000000 !important;
                font-weight: 700 !important;
            }

            /* ---------- Number Input Plus (+) and Minus (-) Buttons: Pure White Background ---------- */
            section[data-testid="stSidebar"] div[data-testid="stNumberInput"] button,
            section[data-testid="stSidebar"] button[data-testid="stNumberInputStepUp"],
            section[data-testid="stSidebar"] button[data-testid="stNumberInputStepDown"],
            section[data-testid="stSidebar"] div[data-testid="stNumberInput"] div[data-baseweb="input"] button {
                background-color: #ffffff !important;
                background: #ffffff !important;
                border: 1px solid #cbd5e1 !important;
                color: #0f172a !important;
                box-shadow: none !important;
                border-radius: 6px !important;
            }
            section[data-testid="stSidebar"] div[data-testid="stNumberInput"] button:hover,
            section[data-testid="stSidebar"] button[data-testid="stNumberInputStepUp"]:hover,
            section[data-testid="stSidebar"] button[data-testid="stNumberInputStepDown"]:hover,
            section[data-testid="stSidebar"] div[data-testid="stNumberInput"] div[data-baseweb="input"] button:hover {
                background-color: #f0f9ff !important;
                background: #f0f9ff !important;
                border-color: #7dd3fc !important;
                color: #0284c7 !important;
            }
            section[data-testid="stSidebar"] div[data-testid="stNumberInput"] button svg,
            section[data-testid="stSidebar"] button[data-testid="stNumberInputStepUp"] svg,
            section[data-testid="stSidebar"] button[data-testid="stNumberInputStepDown"] svg {
                fill: #0f172a !important;
                stroke: #0f172a !important;
                color: #0f172a !important;
            }
            section[data-testid="stSidebar"] div[data-testid="stNumberInput"] button:hover svg,
            section[data-testid="stSidebar"] button[data-testid="stNumberInputStepUp"]:hover svg,
            section[data-testid="stSidebar"] button[data-testid="stNumberInputStepDown"]:hover svg {
                fill: #0284c7 !important;
                stroke: #0284c7 !important;
                color: #0284c7 !important;
            }

            /* Exclude submit button from black text to keep crisp white text on sky blue button */
            section[data-testid="stSidebar"] .stButton > button,
            section[data-testid="stSidebar"] .stButton > button * {
                color: #ffffff !important;
                font-weight: 700 !important;
            }

            /* ---------- Streamlit Buttons ---------- */
            .stButton > button {
                background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
                color: #ffffff !important;
                font-weight: 700 !important;
                border: none !important;
                border-radius: 10px !important;
                padding: 0.55rem 1.1rem !important;
                box-shadow: 0 4px 12px rgba(14, 165, 233, 0.25) !important;
                transition: all 0.2s ease !important;
            }
            .stButton > button:hover {
                transform: translateY(-1px) !important;
                box-shadow: 0 6px 16px rgba(14, 165, 233, 0.35) !important;
                background: linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%) !important;
            }

            /* ---------- Custom Profile Pill Box ---------- */
            .profile-summary-box {
                background: #f0f9ff;
                border: 1px solid #bae6fd;
                border-radius: 12px;
                padding: 0.75rem 1rem;
                margin-bottom: 1.1rem;
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                align-items: center;
            }
            .profile-item {
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                font-size: 0.8rem;
                color: #0369a1;
                background: #ffffff;
                padding: 0.28rem 0.65rem;
                border-radius: 8px;
                border: 1px solid #e0f2fe;
                font-weight: 600;
            }

            hr {
                margin: 1.1rem 0;
                border-color: #f1f5f9;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="header-container">
            <div class="main-header">🌤️ XAI Credit Agent</div>
            <div class="sub-header">
                Sistem Penilaian Risiko Kredit &amp; Otomasi Underwriting Cerdas dengan Interpretasi Matematis dan Rekomendasi AI.
            </div>
            <div>
                <span class="tech-pill">⚡ <b>XGBoost Risk Model</b> (AUC: 0.95)</span>
                <span class="tech-pill">🔍 <b>SHAP TreeExplainer</b> (Explainable AI)</span>
                <span class="tech-pill">🤖 <b>Qwen 2.5 Fine-Tuned LLM</b> (Credit Memo Generator)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# SIDEBAR: APPLICANT INPUT FORM & PRESETS (WITH SOLID BLACK LABELS)
# ==============================================================================
def render_sidebar() -> tuple[ApplicantInput, str, str]:
    st.sidebar.markdown("### 📋 Profil Pemohon Pinjaman")

    preset_option = st.sidebar.selectbox("💡 Pilih Contoh Profil Preset:", list(PRESETS.keys()))
    d = PRESETS[preset_option]

    with st.sidebar.form("applicant_form"):
        st.markdown("**👤 Data Demografi & Finansial**")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Usia (Tahun)", min_value=18, max_value=85, value=d["age"])
            emp_len = st.number_input(
                "Masa Kerja (Thn)", min_value=0.0, max_value=50.0, value=float(d["emp_len"]), step=0.5
            )
        with col2:
            income = st.number_input(
                "Gaji Tahunan ($)", min_value=1000, max_value=2_000_000, value=d["income"], step=5000
            )
            cred_hist = st.number_input("Riwayat Kredit (Thn)", min_value=1, max_value=40, value=d["cred_hist"])

        home_ownership = st.selectbox(
            "Status Rumah",
            HOME_OPTIONS,
            index=HOME_OPTIONS.index(d["home"]),
            format_func=lambda x: HOME_LABELS.get(x, x),
        )

        st.markdown("**💰 Detail Pengajuan Pinjaman**")
        loan_intent = st.selectbox(
            "Tujuan Pinjaman",
            INTENT_OPTIONS,
            index=INTENT_OPTIONS.index(d["intent"]),
            format_func=lambda x: INTENT_LABELS.get(x, x),
        )

        col3, col4 = st.columns(2)
        with col3:
            loan_amnt = st.number_input(
                "Plafon Pinjaman ($)", min_value=500, max_value=100_000, value=d["amount"], step=1000
            )
            loan_grade = st.selectbox("Grade Pinjaman", GRADE_OPTIONS, index=GRADE_OPTIONS.index(d["grade"]))
        with col4:
            loan_int_rate = st.number_input(
                "Suku Bunga (%)", min_value=3.0, max_value=30.0, value=float(d["rate"]), step=0.25
            )
            default_hist = st.selectbox(
                "Pernah Gagal Bayar?",
                ["N", "Y"],
                index=0 if d["default_hist"] == "N" else 1,
                format_func=lambda x: "❌ Ya (Pernah)" if x == "Y" else "✅ Tidak Pernah",
            )

        st.form_submit_button("⚡ Hitung & Evaluasi Risiko", use_container_width=True)

    with st.sidebar.expander("⚙️ Pengaturan Mesin AI (LLM)", expanded=False):
        ollama_endpoint = st.text_input("Ollama Endpoint URL:", value="http://localhost:11434")

        # Deteksi dinamis mesin yang benar-benar aktif di sistem (Ollama, PyTorch, XAI)
        available_engines = discover_available_engines(ollama_endpoint)
        engine_keys = list(available_engines.keys())

        def format_engine_label(key: str) -> str:
            info = available_engines.get(key, {})
            return f"{info.get('label', key)} • {info.get('badge', '')}"

        llm_backend = st.selectbox(
            "Pilih Engine Memo AI:",
            engine_keys,
            index=0,
            format_func=format_engine_label,
            help="Dideteksi secara dinamis: model yang aktif di Ollama (port 11434), bobot PyTorch di models/, dan Fast XAI.",
        )
        selected_engine_info = available_engines.get(llm_backend, {})
        st.caption(f"ℹ️ {selected_engine_info.get('detail', '')}")

    applicant = ApplicantInput(
        age=age, income=income, emp_len=emp_len, home_ownership=home_ownership,
        intent=loan_intent, grade=loan_grade, amount=loan_amnt, rate=loan_int_rate,
        default_hist=default_hist, cred_hist=cred_hist,
    )
    return applicant, llm_backend, ollama_endpoint


# ==============================================================================
# ROW 1: KPI METRICS & RISK GAUGE
# ==============================================================================
def render_applicant_summary_bar(applicant: ApplicantInput) -> None:
    intent_desc = INTENT_LABELS.get(applicant.intent, applicant.intent)
    home_desc = HOME_LABELS.get(applicant.home_ownership, applicant.home_ownership)
    default_desc = "Pernah Default" if applicant.default_hist == "Y" else "Bersih Tanpa Default"

    st.markdown(
        f"""
        <div class="profile-summary-box">
            <span style="font-size: 0.83rem; font-weight: 700; color: #0284c7; margin-right: 0.4rem;">
                👤 Pemohon Aktif:
            </span>
            <span class="profile-item">🎂 Usia: <b>{applicant.age} thn</b></span>
            <span class="profile-item">💼 Kerja: <b>{applicant.emp_len} thn</b></span>
            <span class="profile-item">💵 Gaji: <b>${applicant.income:,.0f}/thn</b></span>
            <span class="profile-item">🏠 Rumah: <b>{home_desc}</b></span>
            <span class="profile-item">🎯 Tujuan: <b>{intent_desc}</b></span>
            <span class="profile-item">🏷️ Grade: <b>{applicant.grade} ({applicant.rate:.2f}%)</b></span>
            <span class="profile-item">🛡️ Rekam Jejak: <b>{default_desc}</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_row(applicant: ApplicantInput, ml_result: dict) -> None:
    pd_val = ml_result["probability_of_default"]
    recommendation = ml_result["recommendation"]
    style = RECOMMENDATION_STYLES.get(recommendation, RECOMMENDATION_STYLES["REJECT"])

    loan_pct_income = (applicant.amount / max(applicant.income, 1)) * 100.0
    is_warning_lti = loan_pct_income > LOAN_TO_INCOME_WARNING_THRESHOLD

    if pd_val < 0.20:
        pd_tier_badge = '<span class="badge badge-approve">🟢 Risiko Rendah (<20%)</span>'
        bar_color = "#10b981"
    elif pd_val <= 0.40:
        pd_tier_badge = '<span class="badge badge-review">🟡 Risiko Moderat (20-40%)</span>'
        bar_color = "#f59e0b"
    else:
        pd_tier_badge = '<span class="badge badge-reject">🔴 Risiko Tinggi (>40%)</span>'
        bar_color = "#ef4444"

    col1, col2, col3, col4 = st.columns([1.3, 1.1, 1.1, 1.2])

    with col1:
        st.markdown(
            f"""
            <div class="decision-hero" style="background-color: {style['bg_color']}; border-color: {style['border_color']};">
                <div class="decision-header" style="color: {style['color']};">Hasil Keputusan Underwriter</div>
                <div class="decision-title" style="color: {style['color']};">{style['badge_icon']} {style['label']}</div>
                <div class="decision-desc" style="color: #475569;">{style['desc']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card metric-card-sky">
                <div>
                    <div class="metric-title">📊 Probabilitas Default (PD)</div>
                    <div class="metric-value" style="color: {style['color']};">{pd_val * 100:.1f}%</div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {min(pd_val * 100, 100):.1f}%; background-color: {bar_color};"></div>
                    </div>
                </div>
                <div style="margin-top: 0.5rem;">{pd_tier_badge}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        lti_badge = (
            '<span class="badge badge-reject">⚠️ Waspada (> 35%)</span>'
            if is_warning_lti
            else '<span class="badge badge-approve">✅ Batas Aman (≤ 35%)</span>'
        )
        total_loan_est = applicant.amount * (1.0 + applicant.rate / 100.0)

        st.markdown(
            f"""
            <div class="metric-card metric-card-sky">
                <div>
                    <div class="metric-title">💳 Rasio Pinjaman / Gaji</div>
                    <div class="metric-value" style="color: {'#dc2626' if is_warning_lti else '#0284c7'};">{loan_pct_income:.1f}%</div>
                    <div class="metric-subtitle">Plafon: <b>${applicant.amount:,.0f}</b> • Beban: <b>${total_loan_est:,.0f}</b></div>
                </div>
                <div style="margin-top: 0.5rem;">{lti_badge}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.plotly_chart(_build_gauge_chart(pd_val, ml_result["risk_color"]), use_container_width=True)


def _build_gauge_chart(pd_val: float, risk_color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pd_val * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            number={"suffix": "%", "font": {"size": 22, "color": "#0f172a", "family": "Plus Jakarta Sans"}},
            title={"text": "Indikator Meter Risiko", "font": {"size": 12, "color": "#64748b"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8", "tickfont": {"size": 10}},
                "bar": {"color": risk_color, "thickness": 0.28},
                "bgcolor": "#f8fafc",
                "borderwidth": 1,
                "bordercolor": "#e2e8f0",
                "steps": [
                    {"range": [0, 20], "color": "rgba(16, 185, 129, 0.25)"},
                    {"range": [20, 40], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [40, 100], "color": "rgba(239, 68, 68, 0.25)"},
                ],
                "threshold": {
                    "line": {"color": "#0284c7", "width": 3},
                    "thickness": 0.8,
                    "value": pd_val * 100,
                },
            },
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=35, b=10),
        height=150,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ==============================================================================
# ROW 2: VARIED PROFILE DATA VISUALIZATIONS (RADAR, DONUT, BENCHMARK)
# ==============================================================================
def _build_radar_profile_chart(applicant: ApplicantInput) -> go.Figure:
    """Radar Chart yang memvisualisasikan 5 pilar kesehatan profil kredit pemohon."""
    loan_pct_income = (applicant.amount / max(applicant.income, 1)) * 100.0

    # Kalkulasi skor 0-100 untuk tiap dimensi
    income_score = min((applicant.income / 100000.0) * 100.0, 100.0)
    emp_score = min((applicant.emp_len / 8.0) * 100.0, 100.0)
    cred_score = min((applicant.cred_hist / 8.0) * 100.0, 100.0)
    lti_capacity = max(100.0 - (loan_pct_income * 2.2), 10.0)

    grade_scores = {"A": 95, "B": 82, "C": 68, "D": 52, "E": 38, "F": 24, "G": 12}
    grade_score = grade_scores.get(applicant.grade, 50)
    if applicant.default_hist == "Y":
        grade_score = max(grade_score - 25, 10)

    categories = [
        "Kapasitas Gaji",
        "Stabilitas Kerja",
        "Rekam Jejak Kredit",
        "Kapasitas Arus Kas",
        "Kualitas Grade Pinjaman",
    ]
    values = [income_score, emp_score, cred_score, lti_capacity, grade_score]
    # Tutup poligon radar
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure(
        go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            fillcolor="rgba(56, 189, 248, 0.22)",
            line=dict(color="#0284c7", width=2.5),
            marker=dict(size=6, color="#0284c7"),
            hoverinfo="theta+r",
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9, color="#64748b")),
            angularaxis=dict(tickfont=dict(size=10, family="Plus Jakarta Sans", color="#1e293b", weight="bold")),
        ),
        showlegend=False,
        margin=dict(l=35, r=35, t=30, b=25),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="🎯 5 Pilar Kesehatan Profil Kredit", font=dict(size=12, color="#0f172a", family="Plus Jakarta Sans")),
    )
    return fig


def _build_financial_donut_chart(applicant: ApplicantInput) -> go.Figure:
    """Donut chart struktur alokasi pendapatan tahunan vs pokok dan estimasi bunga pinjaman."""
    total_interest_est = applicant.amount * (applicant.rate / 100.0)
    total_obligation = applicant.amount + total_interest_est
    remaining_income = max(applicant.income - total_obligation, 0)

    labels = ["Sisa Pendapatan Bersih", "Pokok Pinjaman", "Estimasi Beban Bunga"]
    values = [remaining_income, applicant.amount, total_interest_est]
    colors = ["#10b981", "#0ea5e9", "#f59e0b"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.62,
                marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
                textinfo="percent",
                hoverinfo="label+value+percent",
                textfont=dict(size=11, family="Plus Jakarta Sans"),
            )
        ]
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(size=10)),
        margin=dict(l=15, r=15, t=30, b=35),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="💰 Struktur Finansial & Arus Kas", font=dict(size=12, color="#0f172a", family="Plus Jakarta Sans")),
        annotations=[
            dict(
                text=f"${applicant.income:,.0f}<br><span style='font-size:10px; color:#64748b;'>Total Gaji</span>",
                x=0.5,
                y=0.5,
                font=dict(size=12, color="#0f172a", family="Plus Jakarta Sans", weight="bold"),
                showarrow=False,
            )
        ],
    )
    return fig


def _build_grade_benchmark_chart(applicant: ApplicantInput) -> go.Figure:
    """Bar chart komparasi suku bunga pemohon vs benchmark rata-rata pasar per grade kredit."""
    grades = list(MARKET_GRADE_BENCHMARKS.keys())
    market_rates = list(MARKET_GRADE_BENCHMARKS.values())

    # Warna bar: highlight grade pemohon dengan biru langit mencolok
    bar_colors = [
        "#0284c7" if g == applicant.grade else "rgba(14, 165, 233, 0.25)" for g in grades
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=grades,
            y=market_rates,
            marker=dict(color=bar_colors, line=dict(color="#0284c7", width=1.5)),
            text=[f"{r:.1f}%" for r in market_rates],
            textposition="auto",
            name="Rata-rata Pasar",
            hoverinfo="x+y",
        )
    )

    # Tambahkan garis penanda suku bunga pengajuan aktif pemohon
    fig.add_hline(
        y=applicant.rate,
        line_dash="dot",
        line_color="#ef4444",
        line_width=2,
        annotation_text=f"Bunga Pengajuan: {applicant.rate:.2f}%",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#dc2626"),
    )

    fig.update_layout(
        margin=dict(l=15, r=15, t=30, b=15),
        height=260,
        plot_bgcolor="#ffffff",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans", size=11, color="#334155"),
        xaxis=dict(title="Loan Grade", gridcolor="#f1f5f9"),
        yaxis=dict(title="Suku Bunga (%)", gridcolor="#f1f5f9", range=[0, max(max(market_rates), applicant.rate) + 3]),
        title=dict(text=f"🏷️ Suku Bunga vs Benchmark Pasar (Grade {applicant.grade})", font=dict(size=12, color="#0f172a", family="Plus Jakarta Sans")),
    )
    return fig


def render_profile_visualizations(applicant: ApplicantInput) -> None:
    st.markdown(
        '<div class="section-title">📈 Visualisasi &amp; Analisis Komparatif Profil Pemohon</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1.1, 1.1, 1.2])
    with c1:
        st.plotly_chart(_build_radar_profile_chart(applicant), use_container_width=True)
    with c2:
        st.plotly_chart(_build_financial_donut_chart(applicant), use_container_width=True)
    with c3:
        st.plotly_chart(_build_grade_benchmark_chart(applicant), use_container_width=True)


# ==============================================================================
# ROW 3: EXPLAINABLE AI (SHAP) & AUTOMATED MEMO
# ==============================================================================
def render_shap_panel(ml_result: dict) -> None:
    st.markdown(
        '<div class="section-title">🔍 Explainable AI: Kontribusi Variabel (SHAP)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:0.86rem; color:#64748b; margin-top:-0.4rem;'>"
        "Analisis kontribusi variabel terhadap peningkatan (<span style='color:#ef4444; font-weight:700;'>+ Risiko</span>) "
        "atau peredam (<span style='color:#10b981; font-weight:700;'>- Proteksi</span>) kemungkinan gagal bayar:"
        "</p>",
        unsafe_allow_html=True,
    )

    top_contribs = ml_result["sorted_contributions"][:8]
    feat_df = pd.DataFrame(top_contribs)

    feat_df["feature_label"] = feat_df["feature"].apply(lambda x: FEATURE_NAME_MAP.get(x, x))
    feat_df["type"] = feat_df["shap_value"].apply(
        lambda x: "Pendorong Risiko (+SHAP)" if x > 0 else "Pereda Risiko (-SHAP)"
    )
    feat_df = feat_df.sort_values(by="shap_value", ascending=True)

    fig = px.bar(
        feat_df,
        x="shap_value",
        y="feature_label",
        orientation="h",
        color="type",
        color_discrete_map={
            "Pendorong Risiko (+SHAP)": "#ef4444",
            "Pereda Risiko (-SHAP)": "#10b981",
        },
        labels={"shap_value": "Besaran Pengaruh SHAP", "feature_label": "Variabel Pemohon"},
        title="Top 8 Variabel Paling Berpengaruh (TreeExplainer)",
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=35, b=15),
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""),
        plot_bgcolor="#ffffff",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans", size=12, color="#334155"),
        xaxis=dict(gridcolor="#f1f5f9", zerolinecolor="#cbd5e1", zerolinewidth=1.5),
        yaxis=dict(gridcolor="#f1f5f9"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_memo_panel(raw_input: dict, ml_result: dict, llm_backend: str, ollama_endpoint: str) -> dict:
    header_col, action_col = st.columns([3, 1])
    with header_col:
        st.markdown(
            '<div class="section-title">📄 Automated Underwriting Memo (AI Advisor)</div>',
            unsafe_allow_html=True,
        )
    with action_col:
        force_refresh = st.button("🔄 Generate Ulang", use_container_width=True)

    # Session state caching based on inputs to prevent unnecessary reload freezes
    cache_key = f"memo_{raw_input.get('person_age')}_{raw_input.get('person_income')}_{raw_input.get('loan_amnt')}_{llm_backend}"
    if force_refresh or cache_key not in st.session_state:
        with st.spinner("Menyusun memo pertimbangan kredit dengan AI Underwriter..."):
            try:
                memo_json = generate_credit_memo(raw_input, ml_result, mode=llm_backend, ollama_url=ollama_endpoint)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Gagal membuat memo kredit: {exc}")
                memo_json = {"source": "error", "executive_summary": "Memo tidak dapat dibuat."}
            st.session_state[cache_key] = memo_json
    else:
        memo_json = st.session_state[cache_key]

    engine_source = memo_json.get("source", "XAI LLM Engine")
    st.markdown(
        f"<div style='font-size:0.8rem; color:#0369a1; margin-top:-0.4rem; margin-bottom: 0.6rem;'>"
        f"🤖 Sumber Engine: <span class='badge badge-sky'>{engine_source}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if llm_backend == "ollama" and "Not Ready" in engine_source:
        st.info(
            "💡 **Petunjuk Ollama:** Service Ollama aktif, namun model `qwen2.5:3b` belum terunduh. "
            "Jalankan `ollama run qwen2.5:3b` di terminal untuk menggunakan LLM lokal secara optimal."
        )

    st.markdown(
        f"""
        <div class="card" style="border-left: 4px solid #0ea5e9;">
            <div style="font-weight: 700; color: #0284c7; font-size: 0.92rem; margin-bottom: 0.35rem;">
                📌 Executive Summary
            </div>
            <div style="font-size: 0.9rem; color: #334155; line-height: 1.55;">
                {memo_json.get("executive_summary", "-")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="card" style="border-left: 4px solid #ef4444;">
                <div style="font-weight: 700; color: #dc2626; font-size: 0.88rem; margin-bottom: 0.4rem;">
                    ⚠️ Faktor Pendorong Risiko (+SHAP)
                </div>
            """,
            unsafe_allow_html=True,
        )
        drivers = memo_json.get("key_risk_drivers", [])
        if drivers:
            for item in drivers:
                st.markdown(f"<div style='font-size:0.85rem; color:#475569; margin-bottom:0.3rem;'>• {item}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:0.85rem; color:#94a3b8;'>Tidak ada faktor risiko signifikan.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            """
            <div class="card" style="border-left: 4px solid #10b981;">
                <div style="font-weight: 700; color: #059669; font-size: 0.88rem; margin-bottom: 0.4rem;">
                    🛡️ Faktor Pereda Risiko (-SHAP)
                </div>
            """,
            unsafe_allow_html=True,
        )
        mitigations = memo_json.get("mitigating_factors", [])
        if mitigations:
            for item in mitigations:
                st.markdown(f"<div style='font-size:0.85rem; color:#475569; margin-bottom:0.3rem;'>• {item}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:0.85rem; color:#94a3b8;'>Tidak ada faktor pereda risiko dominan.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    conditions = memo_json.get("conditions_or_mitigation_plan")
    if conditions:
        st.markdown(
            """
            <div class="card" style="border-left: 4px solid #f59e0b;">
                <div style="font-weight: 700; color: #d97706; font-size: 0.88rem; margin-bottom: 0.4rem;">
                    📋 Rekomendasi Syarat Pencairan &amp; Mitigasi
                </div>
            """,
            unsafe_allow_html=True,
        )
        for i, cond in enumerate(conditions, start=1):
            st.markdown(
                f"<div style='font-size:0.85rem; color:#475569; margin-bottom:0.3rem;'><b>{i}.</b> {cond}</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    return memo_json


# ==============================================================================
# ROW 4: AUDIT TRAIL & EXPORT
# ==============================================================================
def render_audit_trail(memo_json: dict, income: int, loan_amnt: int) -> None:
    with st.expander("🛠️ Audit Trail & Raw Data JSON"):
        st.json(memo_json)
        st.download_button(
            label="📥 Unduh Hasil Memo Lengkap (JSON)",
            data=json.dumps(memo_json, indent=2, ensure_ascii=False),
            file_name=f"credit_memo_{int(income)}_{int(loan_amnt)}.json",
            mime="application/json",
        )


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================
def main() -> None:
    configure_page()
    render_header()

    applicant, llm_backend, ollama_endpoint = render_sidebar()
    raw_input = applicant.to_raw_dict()

    df_features = preprocess_applicant_input(raw_input)
    ml_result = predict_credit_risk(df_features)

    # 1. Summary Ribbon of Active Applicant
    render_applicant_summary_bar(applicant)

    # 2. Main KPI Metrics Row & Gauge
    render_kpi_row(applicant, ml_result)

    # 3. Varied Profile Visualizations (Radar, Donut, Benchmark)
    render_profile_visualizations(applicant)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # 4. Two-Column Detailed Analytics & LLM Memo
    col_left, col_right = st.columns([1.15, 1.35])
    with col_left:
        render_shap_panel(ml_result)
    with col_right:
        memo_json = render_memo_panel(raw_input, ml_result, llm_backend, ollama_endpoint)

    # 5. Audit Trail Expander
    render_audit_trail(memo_json, applicant.income, applicant.amount)


if __name__ == "__main__":
    main()