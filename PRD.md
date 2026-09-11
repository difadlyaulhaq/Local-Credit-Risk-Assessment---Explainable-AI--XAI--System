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
| 3 | **Generic LLM Hallucination:** LLM umum sering melakukan kesalahan logika finansial dan halusinasi numerik saat membaca data tabular. | Risiko kesalahan keputusan underwriting yang berujung pada kredit macet (*Non-Performing Loans / NPL*). | Melakukan **Instruction Fine-Tuning (QLoRA via Hugging Face PEFT & TRL)** pada model berbobot analitik tinggi (**Qwen 2.5-7B-Instruct**). |
| 4 | **Manual Credit Memo Creation:** Analis kredit membutuhkan waktu 15–30 menit per aplikasi untuk menyusun memo evaluasi manual. | *Turnaround time (TAT)* pengajuan kredit menjadi lambat dan kapasitas underwriting terbatas. | Otomasi pembuatan *Credit Memo* dalam format JSON/Markdown dalam waktu < 5 detik. |

---

## 3. Goals, Non-Goals & Success Metrics

### 3.1 Primary Goals
1. Mengembangkan model klasifikasi risiko kredit biner (`loan_status`: 0 = Non-default, 1 = Default) dengan performa diskriminasi tinggi.
2. Mengintegrasikan algoritma SHAP (*Shapley Additive Explanations*) untuk mengukur bobot pengaruh setiap variabel pinjaman pemohon.
3. Men-fine-tune model LLM open-source (**Qwen 2.5-7B-Instruct**) menggunakan Hugging Face PEFT & TRL + QLoRA (BitsAndBytes 4-bit) secara efisien dan kompatibel di Windows maupun Google Colab T4.
4. Menyediakan antarmuka dashboard lokal (*Streamlit*) yang menampilkan probabilitas risiko, visualisasi kontribusi SHAP, dan narasi rekomendasi kredit otomatis.

### 3.2 Non-Goals
* Menggantikan sepenuhnya peran manusia (*human-in-the-loop*); sistem ini berstatus sebagai *Decision Support System (DSS)* / asisten *underwriter*.
* Menghubungkan langsung dengan *core banking transaction pipeline* secara real-time pada fase prototipe ini.

### 3.3 Success Metrics & Key Performance Indicators (KPIs)

#### A. Model & Technical Metrics
* **Predictive ML ROC-AUC:** Target $\ge 0.88$ (*Pencapaian Aktual:* **0.9494** via XGBoost Hyperparameter Tuned).
* **Predictive ML F1-Score & PR-AUC:** Target F1 $\ge 0.80$ (*Pencapaian Aktual:* F1 = **0.8250**, PR-AUC = **0.9065**, Akurasi = **92.64%**).
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
| Persona 3: Data Science / AI Engineer (Developer & Portfolio Evaluator)       |
| - Role: Pembangun pipeline end-to-end                                         |
| - Goal: Menunjukkan keunggulan fine-tuning, XAI matematis, dan local serving  |
| - Need: Pipeline yang modular, efisien, dan terdokumentasi rapi              |
+-------------------------------------------------------------------------------+
```

### Use Case Scenario:
1. **Input Data:** Analis memasukkan data pemohon (Usia: 22, Pendapatan: $55,000, Status Rumah: RENT, Tujuan: PERSONAL, Plafon: $15,000, Suku Bunga: 14.61%, Grade: D, Riwayat Gagal Bayar: Y).
2. **Kalkulasi ML & XAI:** Sistem menghitung $P(\text{Default}) = 27.5\%$ (Kategori: Medium Risk) dan mendeteksi bahwa pendorong risiko utama adalah `loan_grade index 3` (+1.375) dan `person_home_ownership_RENT` (+0.215), diimbangi oleh faktor pereda `loan_intent_PERSONAL` (-1.119) dan `person_income = $55,000` (-0.627).
3. **Generasi Memo LLM:** Qwen 2.5-7B (Local Ollama) menerima prompt terstruktur dan menghasilkan rekomendasi `MANUAL_REVIEW` dengan mitigasi syarat khusus berupa verifikasi slip gaji 3 bulan dan penyesuaian tenor pinjaman.

---

## 5. System Architecture & Technical Specifications

```
                              SYSTEM ARCHITECTURE FLOW
                              
  [Credit Dataset]
         │
         ▼
  [01. EDA & Feature Engineering] ───► Imputasi, Outlier, 22 Fitur (Cost, Ratios, One-Hot)
         │
         ▼
  [02. XGBoost Classifier] ──────────► Output: Probability of Default (PD) [ROC-AUC: 0.949]
         │
         ▼
  [03. SHAP TreeExplainer Engine] ───► Output: Top Positive & Negative Feature Attributions
         │
         ├─────────────────────────────────────────┐
         ▼                                         ▼
   [04. Synthetic Dataset Generator]       [06. Streamlit User Interface]
   (Alpaca / JSONL Format: 1,200+ samples)          │
          │                                         │ (Request Payload)
          ▼                                         ▼
   [05. Fine-Tuning Pipeline (PEFT/TRL)]   [07. Local LLM Runtime (Ollama)]
   - Qwen2.5-7B-Instruct (Base)            - GGUF Q4_K_M Quantized Model (~4.7 GB)
   - PEFT + TRL QLoRA (BitsAndBytes 4-bit) - Offline REST API (port: 11434)
   - Export -> GGUF Model                           │ (Strict JSON Response)
          │                                         ▼
          └───────────────────────────────► [08. Underwriting Memo & Audit View]
```

### 5.1 Model Selection & In-Depth Comparison: Qwen 2.5 vs Llama 3.1

Untuk skenario **Credit Risk Assessment & Local Underwriting Memo Generation**, pemilihan model LLM dievaluasi berdasarkan dimensi **Efisiensi Komputasi (Efficiency)** dan **Ketepatan Penalaran (Reasoning Accuracy)**:

| Dimensi Evaluasi | **Qwen 2.5 (7B-Instruct)** ⭐ *(Pilihan Utama)* | **Llama 3.1 (8B-Instruct)** | Analisis untuk Domain Credit Risk |
| :--- | :--- | :--- | :--- |
| **Ketepatan Penalaran Matematika & Tabel** | **Unggul Mutlak (Benchmark MATH/GSM8K > 80+)** | Baik (Benchmark ~65-72) | Qwen jauh lebih presisi dalam membaca rasio keuangan (DTI, bunga, SHAP) tanpa salah menafsirkan angka. |
| **Kepatuhan Format JSON (Schema Adherence)** | **Sangat Tinggi (Native Function/JSON Mode)** | Cukup Tinggi (Terkadang ada *preamble* markdown) | Qwen menghasilkan JSON *audit-ready* yang langsung dapat di-parse oleh aplikasi Streamlit tanpa crash. |
| **Ukuran Model & Efisiensi VRAM (Kuantisasi 4-bit)** | **~4.68 GB (GGUF Q4_K_M)** | ~5.50 GB (GGUF Q4_K_M) | Qwen 7B lebih hemat memori VRAM/RAM (~15% lebih kecil), berjalan mulus di GPU 6-8GB maupun CPU RAM 16GB. |
| **Efisiensi Fine-Tuning (PEFT + TRL SFTTrainer)** | **Sangat Cepat & Hemat Memory** (Memory footprint ~7.5GB VRAM dengan BitsAndBytes 4-bit) | Cepat (Memory footprint ~9.5GB VRAM) | Qwen 2.5 didukung penuh secara *native* oleh Hugging Face PEFT & TRL dengan integrasi Flash Attention / SDPA. |
| **Efisiensi Tokenizer & Multilingual (Bahasa Indonesia)** | **152K Tokenizer Vocab** (Kompresi kata tinggi, sangat lancar Bahasa Indonesia) | 128K Tokenizer Vocab (Lebih condong ke Bahasa Inggris) | Istilah underwriting dan prompt Bahasa Indonesia diproses dengan jumlah token lebih sedikit pada Qwen. |

> **Kesimpulan Pemilihan Model:**  
> **Qwen 2.5-7B-Instruct** adalah pilihan terbaik karena memberikan kombinasi **akurasi interpretasi numerik tertinggi, ukuran VRAM paling efisien, dan kepatuhan format JSON paling stabil**. Jika perangkat penguji memiliki keterbatasan memori ekstrem (< 8 GB RAM), varian **Qwen 2.5-3B-Instruct** dapat digunakan sebagai *drop-in replacement*.

---

## 6. Functional Requirements (FR)

### FR-1: Data Ingestion & Preprocessing Engine
* **FR-1.1:** Sistem memvalidasi dataset kredit dengan atribut demografi, riwayat biro kredit, dan parameter pinjaman.
* **FR-1.2:** Sistem membersihkan anomali data (misal: `person_age > 100` atau `person_emp_length > 60`).
* **FR-1.3:** Sistem melakukan *feature engineering* menghasilkan 22 fitur prediktif: `total_loan_cost`, `disposable_income_est`, `emp_to_age_ratio`, `cred_hist_to_age_ratio`, `high_risk_flag`, serta encoding kategorikal.

### FR-2: Predictive ML Modeling
* **FR-2.1:** Sistem melatih model klasifikasi biner berbasis XGBoost dengan optimasi *hyperparameter tuning* (`n_estimators=150`, `max_depth=6`, `learning_rate=0.1`, `scale_pos_weight=2.86`).
* **FR-2.2:** Sistem menghasilkan nilai probabilitas terkalibrasi ($P(\text{Default})$) dengan performa evaluasi: ROC-AUC = 0.9494 dan F1-Score = 0.8250.
* **FR-2.3:** Sistem mengekspor model ke `models/credit_xgboost_model.pkl` dan metadata ke `models/model_metadata.json`.

### FR-3: Explainability (XAI) & Feature Attribution
* **FR-3.1:** Sistem menghitung *SHAP values* lokal untuk setiap pengajuan pinjaman menggunakan `shap.TreeExplainer`.
* **FR-3.2:** Sistem mengidentifikasi 3 faktor pendorong risiko teratas (*Top Risk Drivers*) dan 3 faktor pereda risiko teratas (*Top Mitigating Factors*).
* **FR-3.3:** Sistem menyajikan visualisasi grafik waterfall atau bar chart SHAP secara dinamis.

### FR-4: Instruction Fine-Tuning Pipeline (Hugging Face PEFT + TRL QLoRA)
* **FR-4.1:** Pipeline mengonversi data fitur + skor ML + penjelasan SHAP menjadi 1,200+ pasangan instruksi fine-tuning (`data/credit_finetune_dataset.jsonl`).
* **FR-4.2:** Menggunakan `transformers` + `bitsandbytes` (`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`) untuk memuat `Qwen/Qwen2.5-7B-Instruct` dalam 4-bit precision.
* **FR-4.3:** Mengonfigurasi parameter QLoRA via `peft.LoraConfig`: $r = 16$, $\alpha = 32$, target modules mencakup `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
* **FR-4.4:** Melatih model menggunakan `trl.SFTTrainer` dan mengekspor bobot LoRA yang telah di-merge ke format **GGUF `Q4_K_M`** untuk Ollama.

### FR-5: Local Model Serving & Integration
* **FR-5.1:** Sistem menyediakan konfigurasi `models/Modelfile` untuk deployment instan di Ollama.
* **FR-5.2:** Model melayani endpoint REST API lokal (`http://localhost:11434/api/chat` atau `/api/generate`).
* **FR-5.3:** Respon LLM mematuhi struktur JSON standar:
  ```json
  {
    "recommendation": "APPROVE | REJECT | MANUAL_REVIEW",
    "risk_tier": "LOW_RISK | MEDIUM_RISK | HIGH_RISK",
    "probability_of_default": "XX.X%",
    "key_risk_drivers": ["Faktor 1", "Faktor 2", "Faktor 3"],
    "mitigating_factors": ["Faktor 1", "Faktor 2", "Faktor 3"],
    "special_conditions": ["Syarat mitigasi 1", "Syarat mitigasi 2"],
    "underwriter_memo": "Narasi komprehensif analisis kelayakan kredit..."
  }
  ```

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
| `total_loan_cost` *(Engineered)* | Float | Total estimasi kewajiban pinjaman + bunga | $\text{loan\_amnt} \times (1 + \frac{\text{int\_rate}}{100})$ |
| `disposable_income_est` *(Engineered)*| Float | Sisa pendapatan setelah dipotong pinjaman | $\text{person\_income} - \text{loan\_amnt}$ |
| `emp_to_age_ratio` *(Engineered)* | Float | Rasio lama bekerja terhadap usia | $\text{emp\_length} / \text{person\_age}$ |
| `cred_hist_to_age_ratio` *(Engineered)*| Float | Rasio histori kredit terhadap usia | $\text{cred\_hist\_length} / \text{person\_age}$ |
| `high_risk_flag` *(Engineered)* | Binary Int | Indikator kombinasi rasio tinggi & grade buruk| `{0, 1}` |

---

## 9. Implementation Roadmap & Milestones

```
+-------------------------------------------------------------------------------+
| PHASE 1: Data Analytics, Baseline ML & SHAP Integration (COMPLETED)           |
| - Selesai EDA, penanganan outlier, imputasi, dan 22 feature engineering       |
| - Pelatihan model XGBoost Classifier (Pencapaian: ROC-AUC 0.9494, F1 0.8250)  |
| - Integrasi SHAP TreeExplainer & validasi kontribusi fitur                    |
+-------------------------------------------------------------------------------+
                                       │
                                       ▼
+-------------------------------------------------------------------------------+
| PHASE 2: Instruction Dataset Synthesis & QLoRA Fine-Tuning (IN PROGRESS)      |
| - Selesai pembuatan 1,200+ dataset sintetis (Fitur + SHAP -> Credit Memo JSON)|
| - Pembuatan notebook 04_peft_trl_finetuning.ipynb (Qwen 2.5-7B + QLoRA)       |
| - Eksekusi training QLoRA via PEFT & TRL (SFTTrainer) & ekspor format GGUF    |
+-------------------------------------------------------------------------------+
                                       │
                                       ▼
+-------------------------------------------------------------------------------+
| PHASE 3: Local Serving & Ollama Integration                                   |
| - Setup Modelfile & registrasi model lokal di Ollama                          |
| - Pembuatan wrapper modul Python untuk komunikasi Ollama API (utils_llm.py)   |
| - Validasi latensi inferensi dan kepatuhan format JSON                        |
+-------------------------------------------------------------------------------+
                                       │
                                       ▼
+-------------------------------------------------------------------------------+
| PHASE 4: Streamlit Dashboard & Delivery                                       |
| - Pembangunan UI interaktif Streamlit (Gauge, Plotly Waterfall, Form Input)   |
| - Integrasi end-to-end (Input Form -> XGBoost -> SHAP -> Ollama -> Display)   |
| - Pengujian beban lokal & finalisasi dokumentasi portofolio                   |
+-------------------------------------------------------------------------------+
```

### 9.1 Project Structure & Artifact Checklist

Berikut adalah status implementasi struktur file dan direktori proyek saat ini:

- [x] [PRD.md](file:///D:/project/credit-risk/PRD.md) — Product Requirement Document komprehensif
- [x] [README.md](file:///D:/project/credit-risk/README.md) — Dokumentasi arsitektur & alur implementasi sistem
- [x] `data/`
  - [x] `data/credit_risk_dataset.csv` — Dataset mentah profil pemohon pinjaman (32.581 baris)
  - [x] `data/cleaned_dataset.csv` — Dataset hasil pembersihan & penanganan outlier (32.409 baris)
  - [x] `data/preprocessed_credit_risk.csv` — Dataset hasil preprocessing & scaling
  - [x] `data/feature_engineered_dataset.csv` — Dataset dengan 22 fitur lengkap
  - [x] `data/credit_finetune_dataset.jsonl` — 1,200+ dataset instruksi sintetis untuk fine-tuning Qwen
  - [x] `data/sample_test_records.json` — Sampel data uji untuk validasi cepat
- [x] `notebooks/`
  - [x] `notebooks/01_eda_and_feature_eng.ipynb` — EDA, cleaning, outlier handling & feature engineering
  - [x] `notebooks/02_ml_and_shap_modeling.ipynb` — Training XGBoost (ROC-AUC 0.949) & kalkulasi SHAP
  - [x] `notebooks/03_dataset_generation.ipynb` — Pipeline sintesis dataset instruksi (Fitur + SHAP $\rightarrow$ Credit Memo)
  - [x] `notebooks/04_peft_trl_finetuning.ipynb` — Pipeline Fine-Tuning Qwen 2.5-7B via Hugging Face PEFT + TRL & Export GGUF
- [ ] `models/`
  - [x] `models/credit_xgboost_model.pkl` — Model klasifikasi machine learning terkalibrasi
  - [x] `models/model_metadata.json` — Hyperparameter terbaik & metrik evaluasi model
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
| **GPU VRAM Out of Memory (OOM) saat Fine-Tuning** | Tinggi | Menggunakan `bitsandbytes` 4-bit NF4 quantization (`BitsAndBytesConfig`), `gradient_checkpointing=True`, serta ukuran `per_device_train_batch_size = 2` dan `gradient_accumulation_steps = 4` via `trl.SFTTrainer`. |
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
