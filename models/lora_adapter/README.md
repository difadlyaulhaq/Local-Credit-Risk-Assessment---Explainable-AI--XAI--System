---
base_model: Qwen/Qwen2.5-3B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-3B-Instruct
- lora
- qlora
- sft
- transformers
- trl
- credit-risk
- explainable-ai
- shap
---

# Qwen 2.5 (3B) - Credit Risk Underwriting & XAI LoRA Adapter

LoRA Adapter ini adalah hasil *Supervised Fine-Tuning (SFT)* pada model **Qwen/Qwen2.5-3B-Instruct** menggunakan teknik **QLoRA (4-bit NF4)**. Model dilatih khusus untuk mengevaluasi data profil pemohon pinjaman, kalkulasi probabilitas gagal bayar (*Probability of Default* dari XGBoost), dan kontribusi faktor risiko matematis (*SHAP Values*), kemudian menghasilkan **Memo Analisis Kredit (*Credit Underwriting Memo*)** terstruktur dalam format JSON valid.

* **Developed by:** Difa Dlyaulhaq
* **Repository:** [https://github.com/difadlyaulhaq/xai-credit-agent](https://github.com/difadlyaulhaq/xai-credit-agent)
* **Base Model:** `Qwen/Qwen2.5-3B-Instruct`
* **Task:** Credit Risk Assessment & Automated Underwriting Memo Generation
* **Language:** Bahasa Indonesia (Formal Banking & Risk Underwriting Terminology)

---

## 🎯 Model Capabilities

Model mampu menghasilkan output JSON dengan skema terstruktur:
* `recommendation`: `APPROVE`, `REJECT`, atau `MANUAL_REVIEW`
* `risk_level`: `LOW_RISK`, `MEDIUM_RISK`, atau `HIGH_RISK`
* `probability_of_default`: Persentase risiko default
* `key_risk_drivers`: Faktor-faktor pendorong risiko terbesar berbasis kontribusi positif (+SHAP)
* `mitigating_factors`: Faktor-faktor pereda risiko berbasis kontribusi negatif (-SHAP)
* `executive_summary`: Narasi profesional pertimbangan kredit bagi *Credit Risk Committee*

---

## ⚙️ Hyperparameter Training

* **PEFT Method:** LoRA (Low-Rank Adaptation)
* **Rank (r):** 8
* **LoRA Alpha:** 16
* **LoRA Dropout:** 0.05
* **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
* **Quantization:** 4-bit NormalFloat (NF4) via BitsAndBytes
* **Optimizer:** Paged AdamW 8-bit
* **Learning Rate:** 2e-4 (Cosine scheduler)
* **Final Training Loss:** 0.1276

---

## 🚀 Cara Penggunaan (Inference Example)

```python
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-3B-Instruct"
adapter_dir = "models/lora_adapter"

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)

# 2. Load Base Model dengan 4-bit Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# 3. Attach LoRA Adapter
model = PeftModel.from_pretrained(base_model, adapter_dir)
model.eval()

# 4. Inferensi ChatML Prompt
prompt_template = """<|im_start|>system
Anda adalah Senior Credit Risk Underwriter AI di institusi perbankan. Tugas Anda adalah mengevaluasi aplikasi kredit pemohon berdasarkan data demografi, keuangan, hasil prediksi model XGBoost (Probability of Default), dan kontribusi faktor risiko matematis (SHAP Values).

Hasilkan laporan analisis kredit (Credit Underwriting Memo) yang terstruktur strictly dalam format JSON valid.<|im_end|>
<|im_start|>user
### PROFIL PEMOHON PINJAMAN:
- Usia Pemohon: 24 tahun
- Pendapatan Tahunan: $45,000
- Status Kepemilikan Rumah: RENT
- Lama Bekerja: 2.0 tahun
- Tujuan Pinjaman: MEDICAL
- Peringkat Risiko Kredit (Grade): C
- Besaran Pinjaman yang Diajukan: $12,000
- Suku Bunga Pinjaman: 13.50%
- Rasio Pinjaman / Pendapatan: 26.7%
- Riwayat Gagal Bayar Sebelumnya: N
- Panjang Riwayat Kredit: 3 tahun

### KALKULASI RISIKO ML & ANALISIS SHAP:
- Prediksi Probability of Default (PD): 31.2%
- Kategori Risiko Awal: MEDIUM_RISK
- Rekomendasi Awal: MANUAL_REVIEW
- Faktor Pendorong Risiko Terbesar (+SHAP):
  * person_home_ownership_RENT bernilai 1.0 (SHAP: +0.28)
  * Rasio pinjaman terhadap pendapatan sebesar 26.7% (SHAP: +0.22)
- Faktor Pereda Risiko Terbesar (-SHAP):
  * Riwayat gagal bayar bersih (cb_person_default_on_file = N) (SHAP: -0.65)
  * Pendapatan tahunan sebesar $45,000 (SHAP: -0.35)<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer([prompt_template], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
json_str = response.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
print(json_str)
```
