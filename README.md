# 🏦 Local Credit Risk Assessment & Explainable AI (XAI) System
### *Automated Credit Memo Generation with Fine-Tuned Local LLMs (Qwen 2.5) & SHAP*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Fine-Tuning](https://img.shields.io/badge/Fine--Tuning-Unsloth%20%2B%20QLoRA-green.svg)](https://github.com/unslothai/unsloth)
[![Model](https://img.shields.io/badge/LLM-Qwen2.5--7B--Instruct-orange.svg)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
[![Serving](https://img.shields.io/badge/Serving-Ollama%20(GGUF%20Q4__K__M)-blueviolet.svg)](https://ollama.ai/)
[![Dashboard](https://img.shields.io/badge/UI-Streamlit-red.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)

---

## 📌 Ringkasan Proyek (Project Overview)

Proyek ini adalah sistem **manajemen dan asesmen risiko kredit (*Credit Risk Management*) *end-to-end*** yang menggabungkan:
1. **Machine Learning Klasik (XGBoost / LightGBM)** untuk kalkulasi probabilitas gagal bayar (*Probability of Default / PD*).
2. **Explainable AI (SHAP)** untuk menganalisis kontribusi setiap fitur finansial pemohon terhadap skor risiko secara transparan dan matematis.
3. **Fine-Tuned Small Language Model (Qwen 2.5-7B-Instruct)** yang dilatih khusus untuk menerjemahkan data tabular & nilai SHAP menjadi **Memo Analisis Kredit (*Credit Underwriting Memo*)** terstruktur (rekomendasi, poin mitigasi, analisis risiko, dan format JSON).
4. **Local & Privacy-Preserving Inference (Ollama / GGUF)** sehingga seluruh data sensitif nasabah (PII) tetap berada di mesin lokal tanpa mengirim API request ke cloud publik.
5. **Interactive Dashboard (Streamlit)** sebagai antarmuka terintegrasi bagi analis kredit (*credit officer* / *underwriter*).

---

## 🏗️ Arsitektur Sistem (System Architecture)

```
+-----------------------------------------------------------------------------------+
|                            1. DATA & PREDICTIVE ML LAYER                          |
|  [Applicant Data] ---> [Preprocessing & Feature Eng] ---> [XGBoost/LightGBM Model]|
|  (Age, Income, Loan,                                       |                      |
|   Home, History, etc.)                                     v                      |
|                                                  [Probability of Default (PD)]    |
+-----------------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
|                        2. EXPLAINABLE AI (XAI) LAYER                              |
|  [SHAP TreeExplainer] ---> Menghitung kontribusi tiap fitur (Feature Importance)  |
|                            Contoh: +loan_percent_income (+0.42), -income (-0.18)  |
+-----------------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
|                     3. FINE-TUNED GENERATIVE AI (LLM) LAYER                       |
|  [Instruction Prompt]                                                             |
|  "Applicant features + SHAP values -> Generate Credit Risk Memo & Recommendation" |
|                                      |                                            |
|                                      v                                            |
|          [Qwen 2.5-7B-Instruct (Fine-Tuned via Unsloth QLoRA, GGUF Q4_K_M)]       |
|                            Served locally via Ollama API                          |
+-----------------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
|                             4. USER INTERFACE LAYER                               |
|  [Streamlit Web App]                                                              |
|   - Form Input Profil Nasabah                                                     |
|   - Gauge Risiko Default & Waterfall Chart SHAP                                  |
|   - Narasi Keputusan Kredit & Rekomendasi Syarat Pinjaman (JSON / Markdown)       |
+-----------------------------------------------------------------------------------+
```

---

## 🎯 Evaluasi & Alasan Pemilihan Model (Model Selection Benchmark)

Untuk skenario **Local Credit Risk Management & Fine-Tuning**, model LLM dipilih berdasarkan 3 kriteria utama:
* **Penalaran Statistik & Matematika (*Mathematical Reasoning*):** Kemampuan menginterpretasi metrik finansial (Debt-to-Income, SHAP values, suku bunga, rasio pendapatan).
* **Efisiensi Ukuran (*Size & VRAM Efficiency*):** Mampu dijalankan secara *offline* di laptop/komputer lokal (RAM 16 GB atau VRAM GPU 6–8 GB).
* **Kepatuhan Instruksi & Output Terformat (*Instruction Following & JSON adherence*):** Menghasilkan laporan analitik yang rapi tanpa halusinasi numerik.

### Matriks Perbandingan Model

| Model LLM | Kategori Ukuran | Skor Math / Reasoning | Kemudahan Local Inference | Rekomendasi Penggunaan |
| :--- | :---: | :---: | :---: | :--- |
| **Qwen 2.5 (7B-Instruct)** ⭐ *(Pilihan Utama)* | **7B** | **Sangat Tinggi (Top Tier)** | **Lancar di GPU 6-8GB / RAM 16GB (Q4)** | **Paling ideal** untuk tugas reasoning numerik, analisis tabel, dan output JSON terstruktur. |
| **Qwen 2.5 (3B-Instruct)** | 3B | Tinggi | Sangat Ringan (RAM 8GB / VRAM 4GB) | Alternatif ultra-ringan jika hardware lokal sangat terbatas. |
| **Llama 3.1 (8B-Instruct)** | 8B | Tinggi | Standar (VRAM 6-8GB) | Sangat bagus untuk narasi bahasa Inggris alami dan integrasi ekosistem luas. |
| **Llama 3.2 (3B-Instruct)** | 3B | Menengah | Sangat Ringan | Cocok untuk prototipe cepat teks umum. |
| **FinMA / FinLlama / Domain Models** | 7B-13B | Variatif | Menengah | Kurang direkomendasikan untuk portofolio karena lebih bernilai menunjukkan kemampuan *fine-tuning* sendiri dari *base model*. |

> **Keputusan Akhir:** Menggunakan **Qwen 2.5 (7B-Instruct)**. Model ini di-fine-tune menggunakan **Unsloth (QLoRA)** di Google Colab (Free T4 GPU), diekspor ke format **GGUF (Q4_K_M)**, dan di-serve di local machine via **Ollama**.

---

## 🛠️ Tech Stack & Hardware Setup

| Komponen | Teknologi / Tools | Keterangan |
| :--- | :--- | :--- |
| **Data & Feature Eng** | `pandas`, `numpy`, `scikit-learn` | Pembersihan data, deteksi outlier, imputasi, dan rasio keuangan. |
| **Predictive ML & XAI** | `xgboost` / `lightgbm`, `shap` | Model klasifikasi default dan *TreeExplainer* untuk interpretabilitas. |
| **LLM Base Model** | `Qwen/Qwen2.5-7B-Instruct` | Model open-source berbobot 7B dengan *reasoning* analitik terbaik. |
| **Fine-Tuning Engine** | `Unsloth` + `QLoRA` (PEFT) | 2x lebih cepat, hemat VRAM 80%, dijalankan di Google Colab GPU T4 (Gratis). |
| **Quantization & Serving**| `GGUF (Q4_K_M)` + `Ollama` | Menjalankan model quantized 4-bit secara lokal dengan latensi rendah. |
| **Application UI** | `Streamlit`, `Plotly` | Dashboard interaktif visualisasi risiko, chart SHAP, dan generator memo. |

---

## 📁 Struktur Direktori Repository

```
credit-risk/
│
├── PRD.md                           # Product Requirements Document & Implementation Checklist
├── README.md                        # Dokumentasi utama proyek & panduan arsitektur
│
├── data/
│   ├── credit_risk_dataset.csv      # Dataset mentah profil pemohon pinjaman
│   ├── cleaned_dataset.csv          # Dataset hasil pembersihan, imputasi KNN & outlier handling
│   └── credit_finetune_dataset.jsonl# Dataset instruksi untuk fine-tuning LLM (Phase 2)
│
├── notebooks/
│   ├── 01_eda_and_feature_eng.ipynb # Pembersihan data, imputasi, outlier handling & EDA
│   ├── 02_ml_and_shap_modeling.ipynb# Training XGBoost + pembuatan SHAP explanations
│   ├── 03_dataset_generation.ipynb  # Pembuatan synthetic instruction dataset (Alpaca/ShareGPT format)
│   └── 04_unsloth_finetuning.ipynb  # Pipeline Fine-Tuning QLoRA di Google Colab & Export GGUF
│
├── models/
│   ├── credit_xgboost_model.pkl     # Model ML tabular biner
│   └── Modelfile                    # Konfigurasi Ollama untuk model GGUF
│
├── app/
│   ├── app.py                       # Main script Streamlit Web Application
│   ├── utils_ml.py                  # Helper inferensi XGBoost & SHAP
│   └── utils_llm.py                 # Client komunikasi Ollama API
│
└── requirements.txt                 # Dependensi pustaka Python
```

---

## 🚀 Alur Eksekusi Proyek (Step-by-Step Pipeline)

### Langkah 1: Exploratory Data Analysis & Feature Engineering
- Menangani nilai hilang (`person_emp_length`, `loan_int_rate`).
- Mengeliminasi outlier data tidak realistis (`person_age > 100`, `person_emp_length > 60`).
- Membuat fitur baru: rasio cicilan, *debt-to-income tier*, dan interaksi histori kredit.

### Langkah 2: Model Klasifikasi & SHAP Interpretation
- Melatih model **XGBoost Classifier** untuk memprediksi `loan_status` (0: Non-default, 1: Default).
- Menggunakan `shap.TreeExplainer` untuk mengekstrak kontribusi marginal (*SHAP values*) setiap fitur untuk setiap individu pemohon pinjaman.

### Langkah 3: Sintesis Dataset Fine-Tuning (Instruction Tuning)
- Mengonversi data tabular + probabilitas default + kontribusi top fitur SHAP menjadi pasangan **Instruction - Input - Output (Credit Risk Memo)**.
- Format Output:
  ```json
  {
    "recommendation": "REJECT / APPROVE / MANUAL_REVIEW",
    "risk_level": "HIGH / MEDIUM / LOW",
    "probability_of_default": "38.5%",
    "key_risk_drivers": [
      "Tingginya rasio pinjaman terhadap pendapatan (loan_percent_income = 0.45)",
      "Adanya catatan riwayat gagal bayar sebelumnya (cb_person_default_on_file = Y)"
    ],
    "mitigating_factors": [
      "Status kepemilikan rumah MORTGAGE menunjukkan stabilitas tempat tinggal"
    ],
    "underwriter_memo": "Pengajuan pinjaman berisiko tinggi karena rasio beban utang melampaui batas toleransi risiko internal (>35%)..."
  }
  ```

### Langkah 4: Fine-Tuning dengan Unsloth & QLoRA di Google Colab
- Memuat model `Qwen/Qwen2.5-7B-Instruct` dengan kuantisasi 4-bit (`bitsandbytes`).
- Mengaplikasikan LoRA adapter pada modul `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- Melatih model selama 2–3 epoch menggunakan `SFTTrainer`.
- Menyimpan hasil adapter dan mengekspor model ter-merge ke format **GGUF `q4_k_m`**.

### Langkah 5: Local Serving via Ollama
1. Pindahkan file model `.gguf` ke laptop/komputer lokal.
2. Buat file `Modelfile`:
   ```dockerfile
   FROM ./qwen2.5-credit-risk-q4_k_m.gguf
   TEMPLATE """{{ if .System }}<|im_start|>system
   {{ .System }}<|im_end|>
   {{ end }}{{ if .Prompt }}<|im_start|>user
   {{ .Prompt }}<|im_end|>
   <|im_start|>assistant
   {{ end }}"""
   PARAMETER temperature 0.2
   PARAMETER top_p 0.9
   PARAMETER stop "<|im_end|>"
   ```
3. Registrasikan ke Ollama:
   ```bash
   ollama create credit-risk-qwen -f Modelfile
   ollama run credit-risk-qwen
   ```

### Langkah 6: Menjalankan Streamlit Dashboard
```bash
# Clone & Navigasi ke direktori
cd D:/project/credit-risk

# Install dependensi
pip install -r requirements.txt

# Jalankan dashboard
streamlit run app/app.py
```

---

## 📊 Keunggulan Nilai Portofolio (Portfolio Value Proposition)

1. **Domain-Specific Fine-Tuning:** Menunjukkan kompetensi praktis dalam membawa *foundation model* umum menjadi spesifik untuk industri *Fintech / Banking Risk*.
2. **Explainable AI (XAI) Integration:** Tidak mengandalkan LLM sebagai *black box*, melainkan menggabungkan nilai SHAP matematis sebagai dasar penalaran (*grounded reasoning*), meminimalisir risiko halusinasi.
3. **Data Privacy & Offline Deployment:** Arsitektur mematuhi standar kerahasiaan data perbankan (tidak mengirim data nasabah ke API pihak ketiga eksternal).
4. **End-to-End Delivery:** Mulai dari eksplorasi data mentah tabular, pemodelan prediktif, *instruction tuning*, kuantisasi GGUF, hingga antarmuka pengguna interaktif.

---

## 📄 Lisensi & Kontribusi
Proyek ini dibuat untuk keperluan riset, portofolio data science, dan edukasi manajemen risiko kredit. Terbuka untuk eksplorasi dan kontribusi lebih lanjut.
