# 🤖 Dokumen 03: Synthetic Instruction Dataset Generation Guide

* **Notebook Rujukan:** `notebooks/03_dataset_generation.ipynb`
* **Input:** `models/credit_xgboost_model.pkl`, `data/preprocessed_credit_risk.csv`, & `data/cleaned_dataset.csv`
* **Output:** `data/credit_finetune_dataset.jsonl` (1.200 baris instruksi terstruktur)

---

## 1. Konsep Desain: Grounded Instruction Synthesis

LLM umum sering mengalami *hallucination* saat membaca angka tabel secara mentah. Untuk mengatasi masalah tersebut, kita menerapkan pendekatan **Grounded Hybrid Generation**:
1. **Fitur Profil Nasabah** (Demografi & Finansial) diterjemahkan ke narasi teks alami.
2. **Kalkulasi Matematis XGBoost** ($P(\text{Default})$) diinjeksikan ke dalam prompt.
3. **Faktor Dominan SHAP** (Top 3 Escalators & Mitigators) diikutsertakan sebagai bukti pendukung (*evidence-based underwriting*).

```
                                  PROMPT FUSION ENGINE
                                  
  [Data Nasabah: Usia 25, Pendapatan $60k, Pinjaman $25k, Rasio 42%]
                                    │
  [Hasil XGBoost: Probability of Default = 34.8% (MEDIUM_RISK)]
                                    │
  [Hasil SHAP: +Rasio Pinjaman (+0.38), +Suku Bunga (+0.25), -Pendapatan (-0.18)]
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │             INSTRUCTION DATASET PAIR (JSONL)            │
       │                                                         │
       │  Instruction: "Anda adalah Senior Credit Underwriter..."│
       │  Input: [Profil Nasabah + Hasil Kalkulasi ML & SHAP]    │
       │  Output: { "recommendation": "MANUAL_REVIEW", ... }     │
       └─────────────────────────────────────────────────────────┘
```

---

## 2. Skema Format Data (Alpaca / ShareGPT Format)

Setiap record dalam `data/credit_finetune_dataset.jsonl` terdiri dari 3 atribut:

### 1. `instruction` (System Persona & Constraints)
Menetapkan peran model sebagai analis kredit senior dengan instruksi ketat untuk menghasilkan JSON memo valid tanpa narasi pembuka/penutup yang tidak perlu.

### 2. `input` (Context Payload)
Rangkuman teks terstruktur yang mencakup:
- Profil demografi & keuangan pemohon pinjaman.
- Hasil probabilitas risiko dari XGBoost.
- Rincian faktor risiko (+SHAP) dan faktor pendukung (-SHAP).

### 3. `output` (Ground-Truth Structured JSON Memo)
```json
{
  "recommendation": "APPROVE | REJECT | MANUAL_REVIEW",
  "risk_tier": "LOW_RISK | MEDIUM_RISK | HIGH_RISK",
  "probability_of_default": "34.8%",
  "key_risk_drivers": [
    "Rasio pinjaman terhadap pendapatan sebesar 42.0%",
    "Suku bunga pinjaman sebesar 16.50%"
  ],
  "mitigating_factors": [
    "Pendapatan tahunan sebesar $60,000",
    "Status kepemilikan rumah MORTGAGE"
  ],
  "special_conditions": [
    "Verifikasi slip gaji dan mutasi rekening 3 bulan terakhir",
    "Penyesuaian jangka waktu tenor untuk menekan angsuran"
  ],
  "underwriter_memo": "Aplikasi pinjaman berada dalam kategori risiko moderat (PD: 34.8%). Diperlukan penyesuaian tenor dan verifikasi berkas tambahan sebelum persetujuan final."
}
```

---

## 3. Strategi Sampling Berimbang (Balanced Sampling)

- Total sampel yang disintesis: **1.200 sampel**.
- **600 sampel Default (Target = 1)** dan **600 sampel Non-Default (Target = 0)**.
- **Tujuan**: Mencegah model LLM menjadi bias (*lazy approval* atau *over-rejection*) saat men-generate rekomendasi kredit.

---

## 4. Validasi Kualitas & Kesiapan Fine-Tuning

Sebelum dataset digunakan pada notebook `04_unsloth_finetuning.ipynb`:
1. **100% Parsing Pass**: Seluruh 1.200 baris output teruji dapat di-parse dengan `json.loads()` tanpa syntax error.
2. **Kepatuhan Kunci**: Semua field (`recommendation`, `risk_tier`, `probability_of_default`, `underwriter_memo`) terisi lengkap.
3. **Format JSONL**: Disimpan dengan encoding `UTF-8` pada `data/credit_finetune_dataset.jsonl`.
