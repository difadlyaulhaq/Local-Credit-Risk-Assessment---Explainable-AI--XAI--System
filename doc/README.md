# 📑 Technical Documentation Index — XAI Credit Agent

Dokumentasi ini menyajikan laporan teknis komprehensif, metodologi langkah demi langkah (*step-by-step methodology*), hasil kuantitatif eksperimen, dan panduan deployment dari seluruh modul pada proyek **XAI Credit Agent**.

* **Repository:** [https://github.com/difadlyaulhaq/xai-credit-agent](https://github.com/difadlyaulhaq/xai-credit-agent)

---

## 🗂️ Daftar Dokumen Teknis

| No | Dokumen | Notebook Rujukan | Topik Utama | Status |
| :---: | :--- | :--- | :--- | :---: |
| **01** | [01_eda_and_feature_engineering_report.md](./01_eda_and_feature_engineering_report.md) | `01_eda_and_feature_eng.ipynb` | Profiling Data, Pembersihan Anomali, Imputasi KNN, Capping Outlier, Rekayasa Fitur Finansial, dan Preprocessing. | ✅ Selesai |
| **02** | [02_predictive_ml_and_shap_xai_report.md](./02_predictive_ml_and_shap_xai_report.md) | `02_ml_and_shap_modeling.ipynb` | Benchmarking 4 Algoritma ML, Penanganan Imbalance, Tuning XGBoost, Evaluasi ROC-AUC (0.9509) & F1 (0.8157), serta Interpretasi SHAP. | ✅ Selesai |
| **03** | [03_synthetic_dataset_generation_guide.md](./03_synthetic_dataset_generation_guide.md) | `03_dataset_generation.ipynb` | Konsep Sintesis Dataset Instruksi (ChatML Prompt), Penyatuan Prediksi ML + SHAP, Skema JSON Underwriting Memo, dan Format JSONL. | ✅ Selesai |
| **04** | [04_unsloth_finetuning_guide.md](./04_unsloth_finetuning_guide.md) | `04_peft_trl_finetuning.ipynb` | Blueprint & Panduan Lengkap Fine-Tuning Qwen 2.5-3B (QLoRA 4-bit NF4) via Hugging Face PEFT + TRL, Evaluasi JSON Schema, dan Penyimpanan Adapter. | ✅ Selesai |

---

## 🏗️ Ringkasan Alur Eksperimen End-to-End

```mermaid
flowchart TD
    A["Raw Dataset (32,581 baris x 12 kolom)"] --> B["01. Data Cleaning & KNN Imputation"]
    B --> C["Winsorization Outliers & Feature Engineering"]
    C --> D["Preprocessed Dataset (32,407 baris x 23 kolom)"]
    D --> E["02. Stratified 5-Fold Benchmarking (XGBoost, LGBM, RF, LogReg)"]
    E --> F["Best Model: XGBoost (ROC-AUC: 0.9509, F1: 0.8157)"]
    F --> G["SHAP TreeExplainer (Feature Attribution)"]
    G --> H["03. Instruction Synthesis Engine (ChatML Schema)"]
    H --> I["credit_finetune_dataset.jsonl (1,200 Balanced Records)"]
    I --> J["04. QLoRA SFT Fine-Tuning Qwen 2.5 (Loss: 0.1276)"]
    J --> K["models/lora_adapter (57.1 MB) -> Streamlit Dashboard"]
```

---

## 🎯 Target Key Performance Indicators (KPI) vs Realisasi

| Metrik / Aspek | Target PRD | Hasil Eksperimen | Status |
| :--- | :---: | :---: | :---: |
| **Predictive ROC-AUC** | >= 0.88 | **0.9509** | ✅ Melampaui Target |
| **Predictive F1-Score** | >= 0.80 | **0.8157** | ✅ Melampaui Target |
| **Precision (Default Class)** | - | **0.8263** | ✅ Sangat Baik |
| **Recall (Default Class)** | - | **0.8054** | ✅ Sangat Baik |
| **Explainability Coverage** | 100% Top Risk Drivers | **100% SHAP Grounded** | ✅ Sesuai Spesifikasi |
| **Instruction JSON Validity** | >= 98% | **100.0% Valid JSON** | ✅ Sempurna |
| **LLM Convergence Loss** | < 0.20 | **0.1276** (Step 90) | ✅ Konvergen Optimal |
