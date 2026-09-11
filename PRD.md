# 📋 Product Requirement Document (PRD)

## Project Title: XAI Credit Agent - Local Explainable Credit Risk Management & Fine-Tuned LLM Underwriting Assistant
* **Repository:** [https://github.com/difadlyaulhaq/xai-credit-agent](https://github.com/difadlyaulhaq/xai-credit-agent)
* **Document Version:** v1.1.0
* **Status:** Phase 1 & 2 Completed / Production Ready
* **Author / Owner:** Difa Dlyaulhaq (Data Science & AI Engineering)
* **Classification:** Internal Product & Open-Source Portfolio Specification

---

## 1. Executive Summary & Product Vision

### 1.1 Executive Summary
Dalam industri perbankan dan *financial technology (fintech)*, evaluasi risiko kredit pemohon pinjaman membutuhkan akurasi statistik yang tinggi sekaligus transparansi keputusan yang dapat diaudit (*audit-ready explainability*). Model Machine Learning tradisional (seperti XGBoost) mampu menghasilkan probabilitas gagal bayar (*Probability of Default / PD*) yang akurat namun sulit dipahami oleh pihak non-teknis tanpa penjelasan kontekstual. Di sisi lain, penggunaan Public Cloud LLM (seperti OpenAI API) menimbulkan kendala kepatuhan privasi data nasabah (*PII data protection*) dan keterbatasan penalaran statistik yang mendalam jika tidak di-fine-tune.

Produk ini menghadirkan **XAI Credit Agent** berbasis *hybrid architecture*: menggabungkan **Predictive ML (XGBoost)**, **Explainable AI (SHAP TreeExplainer)**, dan **Fine-Tuned Local LLM (Qwen 2.5-3B-Instruct / Qwen 2.5-7B-Instruct)** yang berjalan 100% *offline* di lingkungan lokal dan disajikan dalam antarmuka interaktif **Streamlit**.

### 1.2 Product Vision
Menjadi standar platform *credit underwriting* modern yang aman, berbiaya komputasi efisien, dan transparan, yang mampu mengotomasi pembuatan memo persetujuan kredit secara komprehensif dalam hitungan detik tanpa membocorkan data nasabah ke jaringan publik.

---

## 2. Problem Statement & User Pain Points

| No | Pain Point Saat Ini | Dampak Bisnis / Teknis | Solusi yang Diterapkan Produk |
| :--- | :--- | :--- | :--- |
| 1 | **Black-box Decision Making:** Model ML tabular hanya mengeluarkan skor angka (0-1) tanpa narasi alasan terperinci. | *Credit Officer* kesulitan menjelaskan alasan penolakan/persetujuan pinjaman kepada nasabah atau auditor OJK/regulator. | Menggunakan **SHAP** untuk mengekstrak kontribusi fitur, lalu dirangkum menjadi narasi terstruktur oleh LLM. |
| 2 | **Privacy & Regulatory Constraints:** Regulasi perbankan melarang transmisi data sensitif nasabah (PII, pendapatan, riwayat kredit) ke API LLM publik berbasis cloud. | Pembatasan adopsi teknologi Generative AI di institusi finansial. | Menggunakan **Local Inference (Ollama / PyTorch / GGUF)** dengan eksekusi 100% *on-premise / offline* di laptop/server internal. |
| 3 | **Generic LLM Hallucination:** LLM umum sering melakukan kesalahan logika finansial dan halusinasi numerik saat membaca data tabular. | Risiko kesalahan keputusan underwriting yang berujung pada kredit macet (*Non-Performing Loans / NPL*). | Melakukan **Instruction Fine-Tuning (QLoRA via Hugging Face PEFT & TRL)** pada model berbobot analitik tinggi (**Qwen 2.5-3B-Instruct**). |
| 4 | **Manual Credit Memo Creation:** Analis kredit membutuhkan waktu 15–30 menit per aplikasi untuk menyusun memo evaluasi manual. | *Turnaround time (TAT)* pengajuan kredit menjadi lambat dan kapasitas underwriting terbatas. | Otomasi pembuatan *Credit Memo* dalam format JSON/Markdown dalam waktu < 5 detik. |

---

## 3. Goals, Non-Goals & Success Metrics

### 3.1 Primary Goals
1. Mengembangkan model klasifikasi risiko kredit biner (`loan_status`: 0 = Non-default, 1 = Default) dengan performa diskriminasi tinggi.
2. Mengintegrasikan algoritma SHAP (*Shapley Additive Explanations*) untuk mengukur bobot pengaruh setiap variabel pinjaman pemohon.
3. Men-fine-tune model LLM open-source (**Qwen 2.5-3B-Instruct**) menggunakan Hugging Face PEFT & TRL + QLoRA (BitsAndBytes 4-bit) secara efisien dan kompatibel di Windows (GTX 1650 4GB) maupun Google Colab T4.
4. Menyimpan bobot LoRA adapter di `models/lora_adapter/` dan menyediakan dashboard lokal (*Streamlit*) yang menampilkan probabilitas risiko, visualisasi kontribusi SHAP, dan narasi rekomendasi kredit otomatis.

### 3.2 Non-Goals
* Menggantikan sepenuhnya peran manusia (*human-in-the-loop*); sistem ini berstatus sebagai *Decision Support System (DSS)* / asisten *underwriter*.
* Menghubungkan langsung dengan *core banking transaction pipeline* secara real-time pada fase prototipe ini.

### 3.3 Success Metrics & Key Performance Indicators (KPIs)

#### A. Model & Technical Metrics
* **Predictive ML ROC-AUC:** Target >= 0.88 (**Pencapaian Aktual: 0.9509** via XGBoost Tuned).
* **Predictive ML F1-Score:** Target F1 >= 0.80 (**Pencapaian Aktual: 0.8157**, Precision = **0.8263**, Recall = **0.8054**).
* **LLM Training Loss:** **0.1276** (Tercapai pada step 90/300 QLoRA SFT).
* **LLM JSON Schema Adherence:** 100% output LLM mematuhi format skema JSON yang ditentukan tanpa *syntax error*.
* **Local Memory Footprint:** LoRA Adapter berukuran **57.1 MB**, model terkuantisasi 4-bit NF4 muat di VRAM 4GB (GTX 1650).

#### B. Operational & Business Metrics
* **Underwriting Turnaround Time Reduction:** Pengurangan waktu analisis profil risiko sebesar > 60%.
* **Auditability & Traceability:** 100% rekomendasi yang dihasilkan memiliki rujukan langsung ke nilai kalkulasi SHAP (*grounded reasoning*).

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
  [02. XGBoost Classifier] ──────────► Output: Probability of Default (PD) [ROC-AUC: 0.9509]
         │
         ▼
  [03. SHAP TreeExplainer] ──────────► Output: Local & Global Feature Attributions (+/- SHAP)
         │
         ▼
  [04. Synthetic Dataset Generator] ─► 1,200 Balanced Instruction Pairs (credit_finetune_dataset.jsonl)
         │
         ▼
  [05. Fine-Tuning Qwen 2.5 (QLoRA)]─► PEFT + TRL SFTTrainer (Loss: 0.1276) -> models/lora_adapter
         │
         ▼
  [06. Streamlit Web Dashboard] ─────► Interactive Underwriting UI & Real-Time Memo Generation
```

---

## 6. Implementation Milestones

- [x] **Milestone 1: EDA, Imputasi & Rekayasa Fitur** (`01_eda_and_feature_eng.ipynb`)
- [x] **Milestone 2: Modeling Predictive ML & SHAP Interpretability** (`02_ml_and_shap_modeling.ipynb`)
- [x] **Milestone 3: Sintesis Dataset Instruksi ChatML** (`03_dataset_generation.ipynb`)
- [x] **Milestone 4: Fine-Tuning Qwen 2.5 QLoRA via PEFT & TRL** (`04_peft_trl_finetuning.ipynb`)
- [x] **Milestone 5: LoRA Adapter Deployment & Serialization** (`models/lora_adapter`)
- [x] **Milestone 6: Streamlit Web UI & Local Serving** (`app/app.py`)
