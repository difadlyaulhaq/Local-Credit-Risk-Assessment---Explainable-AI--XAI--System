# 📋 Product Requirement Document (PRD)

## Project Title: Local Explainable Credit Risk Management & Fine-Tuned LLM Underwriting Assistant
* **Document Version:** v1.0.0
* **Status:** In Review / Ready for Implementation
* **Author / Owner:** Data Science & AI Engineering Team
* **Target Release:** Q3 2026
* **Classification:** Internal Product & Portfolio Specification

---

## 1. Executive Summary & Product Vision

### 1.1 Executive Summary
Dalam industri perbankan dan *financial technology (fintech)*, evaluasi risiko kredit pemohon pinjaman membutuhkan akurasi statistik yang tinggi sekaligus transparansi keputusan yang dapat diaudit (*audit-ready explainability*). Model Machine Learning tradisional (seperti XGBoost) mampu menghasilkan probabilitas gagal bayar (*Probability of Default / PD*) yang akurat namun sulit dipahami oleh pihak non-teknis tanpa penjelasan kontekstual. Di sisi lain, penggunaan Public Cloud LLM (seperti OpenAI API) menimbulkan kendala kepatuhan privasi data nasabah (*PII data protection*) dan keterbatasan penalaran statistik yang mendalam jika tidak di-fine-tune.

Produk ini menghadirkan **Local Credit Risk Management & Underwriting System** berbasis *hybrid architecture*: menggabungkan **Predictive ML (XGBoost)**, **Explainable AI (SHAP TreeExplainer)**, dan **Fine-Tuned Local LLM (Qwen 2.5-7B-Instruct)** yang berjalan 100% *offline* di lingkungan lokal via **Ollama (GGUF)** dan disajikan dalam antarmuka interaktif **Streamlit**.

### 1.2 Product Vision
Menjadi standar platform *credit underwriting* modern yang aman, berbiaya komputasi efisien, dan transparan, yang mampu mengotomasi pembuatan memo persetujuan kredit secara komprehensif dalam hitungan detik tanpa membocorkan data nasabah ke jaringan publik.

---

## 2. Problem Statement & User Pain Points

| No | Pain Point Saat Ini | Dampak Bisnis / Teknis | Solusi yang Diterapkan Produk |
| :--- | :--- | :--- | :--- |
| 1 | **Black-box Decision Making:** Model ML tabular hanya mengeluarkan skor angka (0-1) tanpa narasi alasan terperinci. | *Credit Officer* kesulitan menjelaskan alasan penolakan/persetujuan pinjaman kepada nasabah atau auditor OJK/regulator. | Menggunakan **SHAP** untuk mengekstrak kontribusi fitur, lalu dirangkum menjadi narasi terstruktur oleh LLM. |
| 2 | **Privacy & Regulatory Constraints:** Regulasi perbankan melarang transmisi data sensitif nasabah (PII, pendapatan, riwayat kredit) ke API LLM publik berbasis cloud. | Pembatasan adopsi teknologi Generative AI di institusi finansial. | Menggunakan **Local Inference (Ollama / GGUF)** dengan eksekusi 100% *on-premise / offline* di laptop/server internal. |
| 3 | **Generic LLM Hallucination:** LLM umum sering melakukan kesalahan logika finansial dan halusinasi numerik saat membaca data tabular. | Risiko kesalahan keputusan underwriting yang berujung pada kredit macet (*Non-Performing Loans / NPL*). | Melakukan **Instruction Fine-Tuning (QLoRA via Unsloth)** pada model berbobot analitik tinggi (**Qwen 2.5-7B-Instruct**). |
| 4 | **Manual Credit Memo Creation:** Analis kredit membutuhkan waktu 15–30 menit per aplikasi untuk menyusun memo evaluasi manual. | *Turnaround time (TAT)* pengajuan kredit menjadi lambat dan kapasitas underwriting terbatas. | Otomasi pembuatan *Credit Memo* dalam format JSON/Markdown dalam waktu < 5 detik. |

---

## 3. Goals, Non-Goals & Success Metrics

### 3.1 Primary Goals
1. Mengembangkan model klasifikasi risiko kredit biner (`loan_status`: 0 = Non-default, 1 = Default) dengan performa diskriminasi tinggi.
2. Mengintegrasikan algoritma SHAP (*Shapley Additive Explanations*) untuk mengukur bobot pengaruh setiap variabel pinjaman pemohon.
3. Men-fine-tune model LLM open-source (**Qwen 2.5-7B-Instruct**) menggunakan Unsloth + QLoRA di Google Colab T4 secara hemat biaya (gratis).
4. Menyediakan antarmuka dashboard lokal (*Streamlit*) yang menampilkan probabilitas risiko, visualisasi kontribusi SHAP, dan narasi rekomendasi kredit otomatis.

### 3.2 Non-Goals
* Menggantikan sepenuhnya peran manusia (*human-in-the-loop*); sistem ini berstatus sebagai *Decision Support System (DSS)* / asisten *underwriter*.
* Menghubungkan langsung dengan *core banking transaction pipeline* secara real-time pada fase prototipe ini.

### 3.3 Success Metrics & Key Performance Indicators (KPIs)

#### A. Model & Technical Metrics
* **Predictive ML ROC-AUC:** $\ge 0.88$ dan F1-Score $\ge 0.80$ pada data pengujian.
* **LLM JSON Schema Adherence:** $\ge 98\%$ output LLM mematuhi format skema JSON yang ditentukan tanpa *syntax error*.
* **Local Inference Latency:** $< 5$ detik per generasi memo pada mesin lokal (GPU VRAM 6-8GB atau RAM 16GB dengan CPU AVX2).
* **Local Memory Footprint:** Model terkuantisasi $\le 5.0\text{ GB}$ (GGUF `Q4_K_M`).

#### B. Operational & Business Metrics
* **Underwriting Turnaround Time Reduction:** Pengurangan waktu analisis profil risiko sebesar $> 60\%$.
* **Auditability & Traceability:** $100\%$ rekomendasi yang dihasilkan memiliki rujukan langsung ke nilai kalkulasi SHAP (*grounded reasoning*).

---

## 4. User Personas & Use Cases

```
+-------------------------------------------------------------------------------+
|                                  USER PERSONAS                                |
+------------------------------------+------------------------------------------+
| Persona 1: Credit Underwriter      | Persona 2: Risk & Compliance Officer     |
| - Role: Reviewer aplikasi kredit   | - Role: Pengawas kepatuhan audit regulasi|
| - Goal: Keputusan cepat & akurat   | - Goal: Memastikan keputusan tidak bias  |
| - Need: Memo ringkas & rekomendasi | - Need: Transparansi metrik SHAP & log   |
+------------------------------------+------------------------------------------+
| Persona 3: Data Science Intern / ML Engineer (Developer)                     |
| - Role: Pembangun pipeline end-to-end                                         |
| - Goal: Menunjukkan kemampuan fine-tuning, XAI, dan local deployment          |
| - Need: Pipeline yang modular, efisien, dan terdokumentasi rapi              |
+-------------------------------------------------------------------------------+
```

### Use Case Scenario:
1. **Input Data:** Analis memasukkan data pemohon (Usia: 25, Pendapatan: $60,000, Status Rumah: RENT, Tujuan: DEBTCONSOLIDATION, Pinjaman: $25,000, Suku Bunga: 16.5%, Riwayat Gagal Bayar: N).
2. **Kalkulasi ML & XAI:** Sistem menghitung $P(\text{Default}) = 34.8\%$ (Kategori: Medium-High Risk) dan mendeteksi bahwa pendorong risiko utama adalah `loan_percent_income = 0.42` (+0.38) dan `loan_int_rate = 16.5%` (+0.25).
3. **Generasi Memo LLM:** Qwen 2.5-7B (Local Ollama) menerima prompt terstruktur dan menghasilkan rekomendasi: `MANUAL_REVIEW / CONDITIONAL_APPROVAL` dengan mitigasi restrukturisasi tenor dan penyesuaian plafon pinjaman.

---

## 5. System Architecture & Technical Specifications

```
                              SYSTEM ARCHITECTURE FLOW
                              
  [Credit Dataset]
         │
         ▼
  [01. EDA & Feature Preprocessing] ───► Imputasi, Deteksi Outlier, Scaling, Encoding
         │
         ▼
  [02. XGBoost / LightGBM Classifier] ──► Output: Probability of Default (PD)
         │
         ▼
  [03. SHAP TreeExplainer Engine] ────► Output: Top Positive & Negative Risk Drivers
         │
         ├─────────────────────────────────────────┐
         ▼                                         ▼
  [04. Synthetic Dataset Generator]       [06. Streamlit User Interface]
  (Alpaca Prompt + SHAP Context)                   │
         │                                         │ (Request / Payload)
         ▼                                         ▼
  [05. Fine-Tuning Pipeline (Colab)]      [07. Local LLM Runtime (Ollama)]
  - Qwen2.5-7B-Instruct (Base)            - GGUF Q4_K_M Quantized Model
  - Unsloth + QLoRA (4-bit)               - Offline Rest API (port: 11434)
  - Export -> GGUF Model                           │ (Structured JSON Response)
         │                                         ▼
         └───────────────────────────────► [08. Underwriting Memo & Audit View]
```

### 5.1 Model Selection Matrix & Architectural Decision Record (ADR)

* **Keputusan:** Memilih **Qwen 2.5 (7B-Instruct)** sebagai model dasar (*Base Model*).
* **Rasional Teknis:**
  1. *Benchmark Reasoning:* Qwen 2.5 secara konsisten melampaui Llama 3.1 pada evaluasi matematika (GSM8K, MATH) dan kemampuan penalaran data terstruktur/tabel.
  2. *Instruction & JSON Formatting:* Memiliki kepatuhan format skema JSON terbukti sangat tinggi, penting untuk integrasi API downstream.
  3. *Efisiensi Kuantisasi:* Varian 7B setelah dikuantisasi ke 4-bit (`Q4_K_M`) hanya berukuran ~4.7 GB, berjalan stabil di VRAM 6-8 GB atau RAM 16 GB tanpa degradasi pemahaman makna yang signifikan.

---

## 6. Functional Requirements (FR)

### FR-1: Data Ingestion & Preprocessing Engine
* **FR-1.1:** Sistem harus mampu memuat dan memvalidasi dataset kredit dengan atribut: `person_age`, `person_income`, `person_home_ownership`, `person_emp_length`, `loan_intent`, `loan_grade`, `loan_amnt`, `loan_int_rate`, `loan_status`, `loan_percent_income`, `cb_person_default_on_file`, `cb_person_cred_hist_length`.
* **FR-1.2:** Sistem harus membersihkan anomali data (contoh: `person_age > 100` atau `person_emp_length > 60`).
* **FR-1.3:** Sistem harus mengimputasi *missing values* numerik menggunakan median per kelompok kategori risiko.

### FR-2: Predictive ML Modeling
* **FR-2.1:** Sistem harus melatih model klasifikasi biner berbasis Gradient Boosting (XGBoost / LightGBM).
* **FR-2.2:** Sistem harus menghasilkan nilai probabilitas terkalibrasi antara 0.00 hingga 1.00 untuk risiko gagal bayar.
* **FR-2.3:** Sistem harus mengekspor model terlatih ke format `.pkl` atau `.joblib`.

### FR-3: Explainability (XAI) & Feature Attribution
* **FR-3.1:** Sistem harus menghitung *SHAP values* lokal untuk setiap pengajuan pinjaman menggunakan `shap.TreeExplainer`.
* **FR-3.2:** Sistem harus mengidentifikasi 3 faktor pendorong risiko teratas (*Top Risk Escalators*) dan 3 faktor pereda risiko teratas (*Top Mitigating Factors*).
* **FR-3.3:** Sistem harus menghasilkan visualisasi grafik waterfall atau bar chart SHAP.

### FR-4: Instruction Fine-Tuning Pipeline (Unsloth + QLoRA)
* **FR-4.1:** Pipeline harus mengonversi data fitur + skor ML + penjelasan SHAP menjadi format instruksi fine-tuning (JSONL format).
* **FR-4.2:** Menggunakan *Unsloth* untuk memuat `Qwen/Qwen2.5-7B-Instruct` dalam 4-bit precision.
* **FR-4.3:** Mengonfigurasi parameter QLoRA: $r = 16$, $\alpha = 32$, target modules mencakup seluruh *attention & MLP projection layers*.
* **FR-4.4:** Mengekspor bobot LoRA yang telah di-merge langsung ke format file **GGUF `Q4_K_M`**.

### FR-5: Local Model Serving & Integration
* **FR-5.1:** Sistem harus menyediakan konfigurasi `Modelfile` untuk deployment instan di Ollama.
* **FR-5.2:** Model harus melayani endpoint REST API lokal (`http://localhost:11434/api/generate` atau `/api/chat`).
* **FR-5.3:** Respon LLM harus mematuhi struktur JSON standar underwriting memo.

### FR-6: Interactive Streamlit Web Interface
* **FR-6.1:** Antarmuka input data profil nasabah yang intuitif dengan validasi batas nilai.
* **FR-6.2:** Tampilan metrik: *Probability of Default Gauge*, *Risk Level Badge*, dan *Decision Recommendation*.
* **FR-6.3:** Visualisasi interaktif kontribusi fitur SHAP menggunakan Plotly.
* **FR-6.4:** Tampilan teks Credit Memo lengkap yang dapat diekspor / disalin (*copy-to-clipboard*).

---

## 7. Non-Functional Requirements (NFR)

| ID | Kategori | Spesifikasi & Kriteria Penerimaan |
| :--- | :--- | :--- |
| **NFR-1** | **Data Privacy & Security** | Seluruh proses inferensi ML dan LLM harus berjalan 100% lokal (*air-gapped capable*). Tidak ada transmisi data nasabah ke server pihak ketiga di luar *local loopback*. |
| **NFR-2** | **Performance & Latency** | Waktu inferensi total (XGBoost + SHAP + LLM) $\le 5$ detik per transaksi pada GPU mid-range (RTX 3060/4060) atau $\le 12$ detik pada CPU modern (8-core RAM 16GB). |
| **NFR-3** | **Hardware Compatibility** | Kebutuhan RAM minimum $\le 16\text{ GB}$ (jika CPU only) atau VRAM GPU $\le 6-8\text{ GB}$. Training dapat dieksekusi di Google Colab T4 (16GB GPU VRAM) tanpa *Out of Memory (OOM)*. |
| **NFR-4** | **Maintainability & Modularity**| Kode terbagi rapi ke dalam modul terpisah: `utils_ml.py`, `utils_llm.py`, dan `app.py` dengan *docstrings* dan tipe anotasi Python lengkap. |
| **NFR-5** | **Reproducibility** | Tersedia *seed* acak (`random_state=42`) di seluruh proses sampling data, modeling, dan fine-tuning untuk memastikan hasil yang dapat direplikasi. |

---

## 8. Data Dictionary & Feature Schema

| Nama Fitur | Tipe Data | Deskripsi | Aturan Validasi |
| :--- | :--- | :--- | :--- |
| `person_age` | Integer | Usia pemohon pinjaman | $18 \le \text{age} \le 80$ |
| `person_income` | Float | Pendapatan tahunan pemohon ($) | $> 0$ |
| `person_home_ownership` | Categorical | Status kepemilikan tempat tinggal | `['RENT', 'MORTGAGE', 'OWN', 'OTHER']` |
| `person_emp_length` | Float | Masa kerja dalam tahun | $0 \le \text{emp\_length} \le 50$ |
| `loan_intent` | Categorical | Tujuan penggunaan pinjaman | `['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT']` |
| `loan_grade` | Categorical | Tingkat risiko awal kredit | `['A', 'B', 'C', 'D', 'E', 'F', 'G']` |
| `loan_amnt` | Float | Jumlah nominal pinjaman yang diajukan ($)| $500 \le \text{loan\_amnt} \le 50,000$ |
| `loan_int_rate` | Float | Suku bunga tahunan (%) | $3.0 \le \text{int\_rate} \le 30.0$ |
| `loan_status` | Binary Int | Target gagal bayar (0: Lunas, 1: Default) | `{0, 1}` |
| `loan_percent_income` | Float | Rasio pinjaman terhadap pendapatan tahunan| $0.0 < \text{ratio} \le 1.0$ |
| `cb_person_default_on_file` | Categorical | Riwayat pernah gagal bayar di biro kredit| `['Y', 'N']` |
| `cb_person_cred_hist_length`| Integer | Lama rekam jejak kredit dalam tahun | $0 \le \text{hist\_length} \le 40$ |

---

## 9. Implementation Roadmap & Milestones

```
+-------------------------------------------------------------------------------+
| PHASE 1: Data Analytics, Baseline ML & SHAP Integration (Week 1)             |
| - Selesai EDA, penanganan outlier, imputasi, dan rekayasa fitur               |
| - Pelatihan model XGBoost Classifier (Target: ROC-AUC >= 0.88)                |
| - Integrasi SHAP TreeExplainer & validasi konsistensi penjelasan              |
+-------------------------------------------------------------------------------+
                                       │
                                       ▼
+-------------------------------------------------------------------------------+
| PHASE 2: Instruction Dataset Synthesis & QLoRA Fine-Tuning (Week 2)          |
| - Pembuatan 1,000+ data sintetis terstruktur (Fitur + SHAP -> Credit Memo)   |
| - Setup lingkungan Google Colab via Unsloth (Qwen2.5-7B-Instruct)             |
| - Eksekusi training QLoRA (Loss convergence < 0.8)                            |
| - Ekspor bobot ke format GGUF Q4_K_M                                          |
+-------------------------------------------------------------------------------+
                                       │
                                       ▼
+-------------------------------------------------------------------------------+
| PHASE 3: Local Serving & Ollama Integration (Week 3)                          |
| - Setup Modelfile & registrasi model lokal di Ollama                          |
| - Pembuatan wrapper modul Python untuk komunikasi Ollama API                  |
| - Validasi latensi inferensi dan kepatuhan format JSON                        |
+-------------------------------------------------------------------------------+
                                       │
                                       ▼
+-------------------------------------------------------------------------------+
| PHASE 4: Streamlit Dashboard & Delivery (Week 4)                              |
| - Pembangunan UI interaktif Streamlit (Gauge, Plotly Waterfall, Form Input)   |
| - Integrasi end-to-end (Input Form -> XGBoost -> SHAP -> Ollama -> Display)   |
| - Pengujian beban lokal & finalisasi dokumentasi portofolio                   |
+-------------------------------------------------------------------------------+
```

### 9.1 Project Structure & Artifact Checklist

Berikut adalah status implementasi struktur file dan direktori proyek:

- [x] `PRD.md` — Product Requirement Document komprehensif
- [x] `README.md` — Dokumentasi arsitektur & alur implementasi sistem
- [x] `data/`
  - [x] `data/credit_risk_dataset.csv` — Dataset mentah profil pemohon pinjaman (32.581 baris)
  - [x] `data/cleaned_dataset.csv` — Dataset hasil pembersihan, imputasi KNN, dan penanganan outlier (32.409 baris)
  - [ ] `data/credit_finetune_dataset.jsonl` — Dataset instruksi sintetis untuk fine-tuning Qwen LLM
- [ ] `notebooks/`
  - [x] `notebooks/01_eda_and_feature_eng.ipynb` — EDA, data cleaning, imputasi missing value & handling outlier
  - [x] `notebooks/02_ml_and_shap_modeling.ipynb` — Training XGBoost/LightGBM & kalkulasi kontribusi SHAP
  - [ ] `notebooks/03_dataset_generation.ipynb` — Pipeline sintesis prompt instruksi (Fitur + SHAP $\rightarrow$ Credit Memo)
  - [ ] `notebooks/04_unsloth_finetuning.ipynb` — Fine-tuning Qwen 2.5-7B via Unsloth (QLoRA) & export GGUF
- [ ] `models/`
  - [x] `models/credit_xgboost_model.pkl` — Model klasifikasi machine learning terkalibrasi
  - [ ] `models/Modelfile` — Konfigurasi serving model GGUF di Ollama
- [ ] `app/`
  - [ ] `app/app.py` — Dashboard UI Streamlit Credit Underwriting Assistant
  - [ ] `app/utils_ml.py` — Helper inferensi XGBoost & kalkulasi visualisasi SHAP
  - [ ] `app/utils_llm.py` — Wrapper client REST API Ollama lokal
- [ ] `requirements.txt` — Daftar dependensi pustaka Python

---

## 10. Risk Management & Mitigation Strategies

| Potensi Risiko | Tingkat Keparahan | Strategi Mitigasi |
| :--- | :---: | :--- |
| **GPU VRAM Out of Memory (OOM) saat Fine-Tuning** | Tinggi | Menggunakan *Unsloth FastLanguageModel* dengan 4-bit quantization, `gradient_checkpointing=True`, serta ukuran `per_device_train_batch_size = 2` dan `gradient_accumulation_steps = 4`. |
| **Halusinasi Nilai Finansial oleh LLM** | Tinggi | Menggunakan pendekatan *grounded generation*: nilai probabilitas default dan faktor risiko terpenting diinjeksikan langsung dari kalkulasi XGBoost dan SHAP ke dalam prompt instruksi secara eksplisit. |
| **Format Output JSON Rusak (*Parsing Error*)** | Sedang | Menetapkan `temperature = 0.1 - 0.2` untuk stabilitas output deterministik, serta menerapkan fungsi *fallback parsing* dengan *regex* di sisi aplikasi. |
| **Latensi Inferensi Lambat pada Perangkat Non-GPU** | Sedang | Menyediakan opsi varian model cadangan ultra-ringan (**Qwen 2.5-3B-Instruct**) untuk perangkat dengan spesifikasi terbatas. |

---

## 11. Approval & Sign-Off

| Stakeholder Role | Nama / Inisial | Status | Tanggal Review |
| :--- | :--- | :--- | :--- |
| **Lead AI Engineer** | Antigravity AI | Approved | 2026-09-01 |
| **Credit Risk Domain Lead** | Reviewer | Approved | 2026-09-01 |
| **Product Manager** | PM Lead | Approved | 2026-09-01 |
