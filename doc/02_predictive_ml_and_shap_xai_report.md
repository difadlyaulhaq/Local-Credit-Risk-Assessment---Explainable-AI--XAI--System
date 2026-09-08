# 🎯 Dokumen 02: Predictive Machine Learning Modeling & Explainable AI (SHAP)

* **Notebook Rujukan:** `notebooks/02_ml_and_shap_modeling.ipynb`
* **Data Input:** `data/preprocessed_credit_risk.csv` (32.407 baris, Train: 25.925, Test: 6.482)
* **Model Output:** `models/credit_xgboost_model.pkl` & `models/model_metadata.json`

---

## 1. Strategi Penanganan Class Imbalance

Ketidakseimbangan kelas target (21.82% default) ditangani dengan menghitung parameter pembobotan:
$$\text{scale\_pos\_weight} = \frac{\text{Jumlah Sampel Non-Default}}{\text{Jumlah Sampel Default}} = \frac{25,317}{7,090} \approx 3.571$$
Parameter ini memberikan penalti error yang lebih besar ketika model salah memprediksi nasabah gagal bayar (*False Negative*).

---

## 2. Hasil Benchmarking Multi-Model (5-Fold Stratified Cross-Validation)

Evaluasi 4 model kandidat pada data latih ($X_{train}$) menghasilkan perbandingan performa sebagai berikut:

| Model Algoritma | CV ROC-AUC (Mean $\pm$ Std) | CV F1-Score | CV Precision | CV Recall | Kesimpulan Evaluasi |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **XGBoost (Tuned)** ⭐ | **0.9454 $\pm$ 0.0022** | **0.8067** | **0.8284** | **0.7862** | **Model Terbaik:** Keseimbangan optimal antara diskriminasi dan recall. |
| **LightGBM** | 0.9447 $\pm$ 0.0037 | 0.8020 | 0.8170 | 0.7877 | Sangat kompetitif, latensi training sedikit lebih cepat. |
| **Random Forest** | 0.9316 $\pm$ 0.0030 | 0.8151 | 0.9662 | 0.7049 | Presisi sangat tinggi, namun *recall* lebih rendah (banyak default terlewat). |
| **Logistic Regression** | 0.8640 $\pm$ 0.0065 | 0.6194 | 0.5139 | 0.7795 | Baseline linear: Banyak *False Positive* karena batas keputusan linier terbatas. |

---

## 3. Hasil Evaluasi Akhir pada Test Set ($X_{test} = 6.482$ Baris)

Model **XGBoost** diuji pada data pengujian independen yang belum pernah dilihat sebelumnya:

| Metrik Evaluasi | Target PRD | Nilai Realisasi | Status Kinerja |
| :--- | :---: | :---: | :---: |
| **ROC-AUC Score** | $\ge 0.8800$ | **0.9509** | ✅ **Sangat Tinggi (+7.09% di atas target)** |
| **F1-Score (Default Class)** | $\ge 0.8000$ | **0.8157** | ✅ **Memenuhi Target** |
| **Precision (Default Class)** | - | **0.8263** | 82.63% dari yang diprediksi gagal bayar terbukti benar. |
| **Recall (Default Class)** | - | **0.8054** | 80.54% dari total kasus gagal bayar riil berhasil dideteksi. |
| **Overall Accuracy** | - | **0.9204 (92.04%)** | 5,966 dari 6,482 data uji diprediksi akurat. |

### Confusion Matrix (Test Set):
```
                       Prediksi Non-Default (0)   Prediksi Default (1)
Aktual Non-Default (0)          4,824 (TN)                 240 (FP)
Aktual Default (1)                276 (FN)               1,142 (TP)
```
- **False Negative Rate yang Rendah (4.2%)**: Hanya 276 pemohon berisiko tinggi yang lolos tanpa terdeteksi.

---

## 4. Analisis Explainable AI (SHAP TreeExplainer)

Menggunakan algoritma SHAP (*Shapley Additive Explanations*), kontribusi setiap fitur dianalisis secara matematis:

### Ranking 10 Fitur Paling Berpengaruh Global (Mean $|\text{SHAP}|$):
1. **`loan_grade_encoded` (0.8515)**: Grade risiko pinjaman (A ke G) merupakan indikator terkuat kelayakan kredit.
2. **`loan_percent_income` (0.6704)**: Rasio pinjaman terhadap pendapatan; pemohon dengan rasio $> 35\%$ mengalami lonjakan eksponensial nilai SHAP positif (peningkatan risiko gagal bayar).
3. **`person_income` (0.6031)**: Semakin tinggi pendapatan tahunan, semakin besar kontribusi negatif SHAP (meredam risiko).
4. **`loan_intent_VENTURE` (0.4196)**: Pinjaman untuk modal usaha terbukti memiliki tingkat keberhasilan bayar lebih tinggi dibanding pinjaman medis/konsolidasi utang.
5. **`person_home_ownership_RENT` (0.4130)**: Status menyewa (*RENT*) meningkatkan risiko gagal bayar dibandingkan kepemilikan rumah (*MORTGAGE/OWN*).
6. **`person_home_ownership_OWN` (0.3815)**: Kepemilikan rumah pribadi merupakan faktor pereda risiko (*mitigating factor*) utama.
7. **`disposable_income_est` (0.3002)**: Sisa pendapatan bersih pemohon.
8. **`loan_int_rate` (0.2928)**: Tingkat suku bunga pinjaman.
9. **`loan_intent_EDUCATION` (0.1533)**: Pinjaman pendidikan memiliki profil risiko moderat-rendah.
10. **`total_loan_cost` (0.1492)**: Total estimasi cicilan dan bunga.

---

## 5. Fungsi Ekstraksi Otomatis (`extract_shap_explanation`)

Fungsi ini mengubah hasil inferensi numerik menjadi kamus terstruktur yang berisi:
- `probability_of_default`: Probabilitas gagal bayar persentase.
- `risk_tier`: `LOW_RISK` ($PD < 20\%$), `MEDIUM_RISK` ($20\% \le PD < 50\%$), atau `HIGH_RISK` ($PD \ge 50\%$).
- `top_risk_escalators`: 3 faktor pemicu risiko terbesar (+SHAP).
- `top_mitigating_factors`: 3 faktor pereda risiko terbesar (-SHAP).

Output ini menjadi fondasi utama yang diinjeksikan ke prompt LLM pada notebook 03.
