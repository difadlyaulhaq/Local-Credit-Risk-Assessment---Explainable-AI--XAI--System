# 📊 Dokumen 01: Exploratory Data Analysis (EDA), Cleaning & Feature Engineering

* **Notebook Rujukan:** `notebooks/01_eda_and_feature_eng.ipynb`
* **Dataset Input:** `data/credit_risk_dataset.csv` (32.581 baris x 12 kolom)
* **Dataset Output:** `data/cleaned_dataset.csv` & `data/preprocessed_credit_risk.csv` (32.407 baris x 23 kolom)

---

## 1. Ringkasan & Profiling Data Awal

Dataset merepresentasikan aplikasi pinjaman individu dengan 11 variabel prediktor dan 1 variabel target:

| Variabel | Tipe Data | Peran / Keterangan | Temuan Awal Data Mentah |
| :--- | :--- | :--- | :--- |
| `person_age` | Numerik | Usia pemohon | Rentang 20 s/d 144 tahun (terdapat anomali usia > 100) |
| `person_income` | Numerik | Pendapatan tahunan ($) | Sangat *right-skewed* ($4,000 s/d $6,000,000) |
| `person_home_ownership`| Kategorikal | Status tempat tinggal | 4 Kategori: `RENT` (50.5%), `MORTGAGE` (41.3%), `OWN` (7.9%), `OTHER` (0.3%) |
| `person_emp_length` | Numerik | Masa kerja (tahun) | Missing: 895 baris (2.75%), anomali: 123 tahun |
| `loan_intent` | Kategorikal | Tujuan pinjaman | 6 Kategori: `EDUCATION`, `MEDICAL`, `VENTURE`, `PERSONAL`, `DEBTCONSOLIDATION`, `HOMEIMPROVEMENT` |
| `loan_grade` | Ordinal | Grade risiko kredit | 7 Tingkat: `A` (terbaik) hingga `G` (terburuk) |
| `loan_amnt` | Numerik | Nominal pinjaman ($) | Median $8,000 (Rentang $500 s/d $35,000) |
| `loan_int_rate` | Numerik | Suku bunga (%) | Missing: 3.116 baris (9.56%), Rentang 5.42% s/d 23.22% |
| `loan_percent_income` | Numerik | Rasio pinjaman/pendapatan | Rentang 0.00 s/d 0.83 (Median: 0.15) |
| `cb_person_default_on_file` | Kategorikal | Riwayat gagal bayar | `Y` (Pernah: 17.6%), `N` (Tidak pernah: 82.4%) |
| `cb_person_cred_hist_length`| Numerik | Lama riwayat kredit (thn)| Median: 4 tahun (Rentang 2 s/d 30 tahun) |
| **`loan_status`** | **Target (Biner)** | **Status kredit pemohon** | **0 = Non-Default (78.18%), 1 = Default (21.82%)** |

### Temuan Utama Distribusi Target:
- Rasio ketidakseimbangan kelas (*imbalance ratio*) adalah **1 : 3.58** (21.82% default).
- Ini mengindikasikan bahwa evaluasi model tidak boleh semata-mata mengandalkan *Accuracy*, melainkan wajib menggunakan **ROC-AUC**, **F1-Score**, dan **PR-AUC**.

---

## 2. Metodologi Langkah Pembersihan Data (Data Cleaning)

```
[Raw Data: 32,581 baris]
        │
        ▼
[1. Hapus Duplikat Identik] ──► -165 baris duplikat (Sisa: 32,416)
        │
        ▼
[2. Hapus Data Entry Error] ──► -7 baris (Age > 100 thn & Emp Length > 100 thn) (Sisa: 32,409)
        │
        ▼
[3. Imputasi KNN & Modus]   ──► KNNImputer(k=5) untuk int_rate & emp_length
        │
        ▼
[4. Outlier Capping (P99)]  ──► Winsorization pada batas persentil 99 untuk income & emp_length
        │
        ▼
[Cleaned Data: 32,409 baris]
```

### Analisis & Rationale Keputusan Teknis:
1. **Mengapa KNN Imputer, bukan Mean/Median Imputation?**
   - Nilai `loan_int_rate` berkorelasi sangat kuat dengan `loan_grade` dan `loan_amnt`. Imputasi nilai rata-rata sederhana akan merusak struktur multivariat. `KNNImputer(n_neighbors=5)` memperhitungkan kedekatan jarak fitur lain untuk mengestimasi suku bunga secara presisi.
2. **Mengapa Winsorization / Capping (P99) bukan Trimming?**
   - Menghapus seluruh data outlier secara agresif (*trimming*) akan menghilangkan >15% baris data nasabah valid berpendapatan tinggi. Capping pada persentil 99 meredam dampak nilai ekstrem terhadap model tanpa kehilangan ukuran sampel data.

---

## 3. Rekayasa Fitur (Feature Engineering)

Untuk memperkaya representasi risiko kredit perbankan, ditambahkan 5 fitur rasio domain finansial:

| Fitur Baru | Formula Matematis | Rasional & Analisis Domain Risiko |
| :--- | :--- | :--- |
| `total_loan_cost` | $\text{loan\_amnt} \times (1 + \frac{\text{loan\_int\_rate}}{100})$ | Mengukur total kewajiban finansial riil (pokok + bunga) yang harus dilunasi pemohon. |
| `disposable_income_est` | $\text{person\_income} - \text{loan\_amnt}$ | Proxy daya tahan finansial pemohon setelah menyisihkan dana pinjaman. |
| `emp_to_age_ratio` | $\frac{\text{person\_emp\_length}}{\text{person\_age}}$ | Proxy stabilitas karier dan loyalitas kerja terhadap umur biologis. |
| `cred_hist_to_age_ratio` | $\frac{\text{cb\_person\_cred\_hist\_length}}{\text{person\_age}}$ | Mengukur seberapa awal nasabah mulai terekspos dengan sistem perbankan. |
| `high_risk_flag` | $(\text{default}=\text{'Y'}) \land (\text{ratio} > 0.35)$ | Indikator risiko tinggi gabungan (*prior default* + beban utang melampaui batas aman 35%). |

---

## 4. Preprocessing & Encoding

1. **Ordinal Encoding**:
   - `loan_grade`: Dimetakan dari `A` (0, risiko terendah) hingga `G` (6, risiko tertinggi).
   - `cb_person_default_on_file`: Dimetakan `N` = 0, `Y` = 1.
2. **One-Hot Encoding**:
   - `person_home_ownership` (`MORTGAGE`, `OWN`, `RENT`, `OTHER` dengan `drop_first=True`).
   - `loan_intent` (`EDUCATION`, `HOMEIMPROVEMENT`, `MEDICAL`, `PERSONAL`, `VENTURE`, `DEBTCONSOLIDATION`).
3. **Hasil Akhir**:
   - Dataset numerik utuh berukuran **32.407 baris x 23 kolom**.
   - Disimpan ke `data/preprocessed_credit_risk.csv`.
