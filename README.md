# 🏦 XAI Credit Agent: Explainable AI & Agentic Credit Risk Underwriting System
### *Automated Credit Memo Generation with Fine-Tuned Qwen 2.5 (QLoRA), XGBoost & SHAP*

[![GitHub Repository](https://img.shields.io/badge/GitHub-difadlyaulhaq%2Fxai--credit--agent-blue.svg?logo=github)](https://github.com/difadlyaulhaq/xai-credit-agent)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Model](https://img.shields.io/badge/LLM-Qwen2.5--3B--Instruct-orange.svg)](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
[![Fine-Tuning](https://img.shields.io/badge/Fine--Tuning-PEFT%20%2B%20TRL%20(QLoRA)-green.svg)](https://github.com/huggingface/peft)
[![Serving](https://img.shields.io/badge/Serving-Ollama%20%2F%20vLLM%20(GGUF%20%2F%20Safetensors)-blueviolet.svg)](https://ollama.ai/)
[![Dashboard](https://img.shields.io/badge/UI-Streamlit-red.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)

---

## 📌 Ringkasan Proyek (Project Overview)

**XAI Credit Agent** adalah sistem manajemen dan asesmen risiko kredit (*Credit Risk Management*) *end-to-end* yang menggabungkan:
1. **Machine Learning Klasik (XGBoost / LightGBM)** untuk kalkulasi probabilitas gagal bayar (*Probability of Default / PD*) dengan performa tinggi (**ROC-AUC: 0.9509**, **F1-Score: 0.8157**).
2. **Explainable AI (SHAP TreeExplainer)** untuk mengekstrak kontribusi matematis tiap fitur finansial pemohon (*Feature Attribution* / *Risk Drivers*) secara transparan.
3. **Fine-Tuned Generative LLM (Qwen 2.5-3B-Instruct)** yang dilatih khusus menggunakan **QLoRA (4-bit NF4)** untuk mengonversi data pemohon + metrik SHAP menjadi **Memo Analisis Kredit (*Credit Underwriting Memo*)** berformat JSON terstruktur dan audit-ready.
4. **Local & Privacy-Preserving Inference (Ollama / Local PyTorch)** sehingga data sensitif nasabah (PII) tetap aman di infrastruktur lokal tanpa dikirim ke cloud pihak ketiga.
5. **Interactive Dashboard (Streamlit)** sebagai antarmuka terintegrasi bagi *credit risk officer* dan *underwriter*.

---

## 🏗️ Arsitektur Sistem (System Architecture)

```
+-----------------------------------------------------------------------------------+
|                            1. DATA & PREDICTIVE ML LAYER                          |
|  [Applicant Data] ---> [Preprocessing & Feature Eng] ---> [XGBoost Model (0.95 AUC]|
|  (Age, Income, Loan,                                       |                      |
|   Home, History, etc.)                                     v                      |
|                                                  [Probability of Default (PD)]    |
+-----------------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
|                        2. EXPLAINABLE AI (XAI) LAYER                              |
|  [SHAP TreeExplainer] ---> Menghitung kontribusi tiap fitur (Feature Attribution) |
|                            Contoh: +loan_percent_income (+0.42), -income (-0.18)  |
+-----------------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
|                     3. FINE-TUNED GENERATIVE AI (LLM) LAYER                       |
|  [ChatML Prompt]                                                                  |
|  "Applicant features + SHAP values -> Generate Credit Risk Memo & Recommendation" |
|                                      |                                            |
|                                      v                                            |
|          [Qwen 2.5-3B-Instruct (Fine-Tuned via QLoRA 4-bit NF4)]                 |
|                   LoRA Adapter: models/lora_adapter (57.1 MB)                     |
+-----------------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
|                             4. USER INTERFACE LAYER                               |
|  [Streamlit Web Application]                                                      |
|   - Form Input Profil Nasabah                                                     |
|   - Gauge Risiko Default & Waterfall Chart SHAP                                  |
|   - Narasi Keputusan Kredit & Rekomendasi Syarat Pinjaman (JSON / Markdown)       |
+-----------------------------------------------------------------------------------+
```

---

## 🎯 Model Benchmark & Decision Matrix

| Model LLM | Ukuran | Kemampuan Analisis Finansial | Hardware Requirement | Status di Proyek |
| :--- | :---: | :---: | :---: | :--- |
| **Qwen 2.5 (3B-Instruct)** ⭐ | **3B** | **Sangat Tinggi (Top Tier)** | **Lancar di GPU 4GB VRAM / RAM 8GB** | **Model Utama (Trained & Deployed)** |
| **Qwen 2.5 (7B-Instruct)** | 7B | Sangat Tinggi | GPU 8GB–16GB VRAM / Colab T4 | Opsi Skala Besar |
| **Llama 3.1 (8B-Instruct)** | 8B | Tinggi | GPU 8GB–16GB VRAM | Alternatif Bahasa Inggris |
| **XGBoost (Tabular Classifier)**| - | **ROC-AUC: 0.9509** | CPU / GPU Ringan | **Model Predictive Default Utama** |

> **Implementasi Fine-Tuning**: Model dilatih menggunakan teknik **QLoRA (4-bit NF4)** pada dataset `data/credit_finetune_dataset.jsonl` (1.200 sampel berpasangan). Bobot adapter tersimpan di [`models/lora_adapter/`](./models/lora_adapter/).

---

## 🛠️ Tech Stack

| Komponen | Teknologi | Keterangan |
| :--- | :--- | :--- |
| **Data & Feature Engineering** | `pandas`, `numpy`, `scikit-learn` | Imputasi KNN, outlier capping (P99), rekayasa rasio finansial. |
| **Predictive ML & XAI** | `xgboost`, `shap` | Model klasifikasi default & *TreeExplainer* interpretabilitas. |
| **LLM Foundation Model** | `Qwen/Qwen2.5-3B-Instruct` | Large Language Model dengan reasoning analitik kuat. |
| **Fine-Tuning Stack** | `transformers`, `peft`, `trl`, `bitsandbytes` | QLoRA 4-bit NormalFloat, SFTConfig, dan Paged AdamW 8-bit. |
| **Serving & Inference** | `PyTorch`, `Ollama` / `GGUF` | Inferensi lokal aman dan cepat. |
| **Application UI** | `Streamlit`, `Plotly`, `Seaborn` | Dashboard interaktif visualisasi risiko & underwriter memo. |

---

## 📁 Struktur Direktori Repository

```
xai-credit-agent/
│
├── PRD.md                           # Product Requirements Document & Tech Specs
├── README.md                        # Dokumentasi utama proyek
├── requirements.txt                 # Daftar dependensi Python resmi
│
├── data/
│   ├── credit_risk_dataset.csv      # Dataset mentah profil pemohon pinjaman
│   ├── cleaned_dataset.csv          # Dataset hasil pembersihan, imputasi KNN & outlier handling
│   ├── preprocessed_credit_risk.csv # Dataset hasil feature engineering
│   ├── credit_finetune_dataset.jsonl# Dataset instruksi fine-tuning LLM (1.200 sampel)
│   └── sample_test_records.json     # Contoh data uji profil pemohon
│
├── notebooks/
│   ├── 01_eda_and_feature_eng.ipynb # Pembersihan data, imputasi, outlier handling & EDA
│   ├── 02_ml_and_shap_modeling.ipynb# Training XGBoost + pembuatan SHAP explanations
│   ├── 03_dataset_generation.ipynb  # Pembuatan synthetic instruction dataset (ChatML schema)
│   └── 04_peft_trl_finetuning.ipynb # Pipeline Fine-Tuning QLoRA (4-bit NF4) & validasi JSON
│
├── models/
│   ├── credit_xgboost_model.pkl     # Bobot model Machine Learning XGBoost
│   ├── model_metadata.json          # Metadata fitur & threshold klasifikasi
│   └── lora_adapter/                # Bobot LoRA Adapter fine-tuned Qwen 2.5 (57.1 MB)
│       ├── adapter_model.safetensors# Bobot adapter LoRA
│       ├── adapter_config.json      # Konfigurasi rank & target modules
│       └── tokenizer*               # Tokenizer & chat template Jinja
│
├── doc/
│   ├── README.md                    # Indeks dokumentasi teknis
│   ├── 01_eda_and_feature_engineering_report.md
│   ├── 02_predictive_ml_and_shap_xai_report.md
│   ├── 03_synthetic_dataset_generation_guide.md
│   └── 04_unsloth_finetuning_guide.md
│
└── app/
    ├── app.py                       # Main script Streamlit Web Application
    ├── utils_ml.py                  # Helper inferensi XGBoost & kalkulasi SHAP
    └── utils_llm.py                 # Helper pemanggilan model LLM / Ollama
```

---

## 🚀 Panduan Menjalankan Sistem (Getting Started)

### 1. Kloning Repositori & Persiapan Lingkungan

```bash
git clone https://github.com/difadlyaulhaq/xai-credit-agent.git
cd xai-credit-agent

# Buat dan aktifkan virtual environment (opsional tapi disarankan)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install seluruh dependensi
pip install -r requirements.txt
```

### 2. Menjalankan Eksperimen di Notebooks
Notebook dapat dijalankan berurutan:
1. `notebooks/01_eda_and_feature_eng.ipynb` -> Pembersihan dan feature engineering data tabular.
2. `notebooks/02_ml_and_shap_modeling.ipynb` -> Pelatihan model XGBoost dan ekstraksi nilai SHAP.
3. `notebooks/03_dataset_generation.ipynb` -> Pembuatan dataset JSONL instruksi underwriting.
4. `notebooks/04_peft_trl_finetuning.ipynb` -> Fine-tuning model LLM Qwen 2.5 dengan QLoRA 4-bit.

### 3. Menguji Inferensi Model Fine-Tuned (Python Script)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-3B-Instruct"
adapter_dir = "models/lora_adapter"

tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True
)
model = PeftModel.from_pretrained(base_model, adapter_dir)

print("Model XAI Credit Agent siap digunakan untuk inferensi!")
```

### 4. Menjalankan Dashboard Streamlit

```bash
streamlit run app/app.py
```

---

## 📄 Contoh Output Format JSON Underwriting Memo

```json
{
  "recommendation": "MANUAL_REVIEW",
  "risk_level": "MEDIUM_RISK",
  "probability_of_default": "31.2%",
  "key_risk_drivers": [
    "Status kepemilikan rumah RENT memberikan kontribusi peningkatan risiko (SHAP: +0.28)",
    "Rasio pinjaman terhadap pendapatan sebesar 26.7% mendekati ambang batas konservatif (SHAP: +0.22)"
  ],
  "mitigating_factors": [
    "Riwayat kredit bersih tanpa catatan gagal bayar sebelumnya (SHAP: -0.65)",
    "Tingkat pendapatan tahunan stabil di angka $45,000 (SHAP: -0.35)"
  ],
  "executive_summary": "Aplikasi kredit dikategorikan sebagai risiko moderat. Profil finansial pemohon menunjukkan stabilitas pendapatan dan riwayat kredit yang sangat baik, namun status tempat tinggal sewa dan rasio pinjaman yang cukup tinggi memerlukan verifikasi tambahan atas arus kas bulanan."
}
```

---

## 👤 Author
* **Difa Dlyaulhaq** — [GitHub Profile](https://github.com/difadlyaulhaq) | [Repository](https://github.com/difadlyaulhaq/xai-credit-agent)
