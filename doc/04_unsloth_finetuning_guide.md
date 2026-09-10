# 🚀 Dokumen 04: Blueprint & Panduan Implementasi Fine-Tuning Qwen 2.5 dengan Hugging Face (PEFT + TRL + BitsAndBytes QLoRA)

* **Target Notebook:** `notebooks/04_peft_trl_finetuning.ipynb`
* **Target Komputasi:** Google Colab (Free T4 GPU / VRAM 15-16 GB) atau Local GPU (VRAM $\ge 8$ GB) / Windows Native
* **Base Model:** `Qwen/Qwen2.5-7B-Instruct` (atau varian 3B: `Qwen/Qwen2.5-3B-Instruct`)
* **Input File:** `data/credit_finetune_dataset.jsonl` (1.200 sampel)
* **Output File:**
  - `models/lora_adapter/` (LoRA adapter weights & tokenizer)
  - `models/merged_model/` (Merged full weights untuk konversi GGUF)
  - `models/qwen2.5-credit-risk-q4_k_m.gguf` (Model terkuantisasi 4-bit)
  - `models/Modelfile` (Konfigurasi serving Ollama)

---

## 📌 Alur Pipeline Fine-Tuning (PEFT + TRL)

```
                                      FINE-TUNING WORKFLOW
                                      
  [data/credit_finetune_dataset.jsonl]
                   │
                   ▼
  [1. Environment Setup] ─────────────► Install transformers, peft, trl, bitsandbytes, accelerate, datasets
                   │
                   ▼
  [2. Load Base Model (4-bit NF4)] ───► AutoModelForCausalLM + BitsAndBytesConfig (4-bit NF4)
                   │
                   ▼
  [3. Attach QLoRA Adapters] ─────────► LoraConfig (r=16, alpha=32, target_modules=All Projections)
                   │
                   ▼
  [4. ChatML Formatting & Tokenize] ──► <|im_start|>system...<|im_start|>user...<|im_start|>assistant...
                   │
                   ▼
  [5. SFT Training Execution] ────────► TRL SFTTrainer (Batch Size 2, Grad Accum 4, AdamW 8-bit)
                   │
                   ▼
  [6. Validation & JSON Parsing Test] ► Evaluasi Inferensi & Validasi Skema JSON Memo
                   │
                   ▼
  [7. Merge Adapter & Export GGUF] ──► Merge LoRA + Base Model -> Konversi GGUF Q4_K_M untuk Ollama
                   │
                   ▼
  [8. Generate Modelfile] ────────────► Template & parameter serving Ollama lokal
```

---

## 🛠️ Rincian Sel / Blok Kode (Step-by-Step Implementation Guide)

Berikut adalah struktur terperinci tiap sel yang dapat kamu tulis dan jalankan di notebook:

---

### Bagian 1: Instalasi Dependensi & Verifikasi GPU

#### Tujuan:
Memasang dependensi resmi Hugging Face yang kompatibel lintas platform (Windows native dan Google Colab) tanpa bergantung pada Triton khusus Linux.

#### Kode Sel:
```python
# Sel 1: Install Hugging Face PEFT + TRL Stack (Kompatibel Windows & Colab)
!pip install --upgrade pip
!pip install transformers peft bitsandbytes accelerate datasets trl scipy

# Sel 2: Verifikasi GPU & Alokasi VRAM
import torch

if torch.cuda.is_available():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"✅ GPU Terdeteksi: {gpu_stats.name} ({max_memory} GB)")
    print(f"Initial VRAM Reserved: {start_gpu_memory} GB")
else:
    print("⚠️ GPU CUDA tidak terdeteksi! Pastikan runtime GPU aktif.")
```

---

### Bagian 2: Memuat Base Model & Tokenizer (4-bit NF4 Quantization)

#### Tujuan:
Memuat model `Qwen/Qwen2.5-7B-Instruct` dalam presisi 4-bit NF4 menggunakan `BitsAndBytesConfig` dan `prepare_model_for_kbit_training` dari PEFT agar hemat VRAM ($\sim 6-7\text{ GB}$).

#### Kode Sel:
```python
# Sel 3: Inisialisasi BitsAndBytesConfig, Model & Tokenizer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

model_id = "Qwen/Qwen2.5-7B-Instruct"
# Opsi alternatif jika VRAM sangat terbatas: "Qwen/Qwen2.5-3B-Instruct"

# Konfigurasi 4-bit NF4 Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    padding_side="right"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load Base Model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Persiapkan model untuk k-bit training (freeze base weights + enable gradient checkpointing)
model = prepare_model_for_kbit_training(model)
print("✅ Base Model berhasil dimuat dalam 4-bit NF4!")
```

---

### Bagian 3: Konfigurasi Adapter QLoRA (PEFT)

#### Tujuan:
Menyematkan adapter LoRA pada seluruh layer proyeksi *attention* dan *feed-forward network (MLP)*.

#### Spesifikasi Hyperparameter LoRA:
* `r` (Rank): `16`
* `lora_alpha`: `32` (Rasio $\alpha/r = 2$)
* `lora_dropout`: `0.05`
* `target_modules`: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
* `task_type`: `"CAUSAL_LM"`

#### Kode Sel:
```python
# Sel 4: Setup LoraConfig & Get PEFT Model
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

---

### Bagian 4: Memuat Dataset & Pemformatan Prompt ChatML

#### Tujuan:
Mengonversi format JSONL (`instruction`, `input`, `output`) ke struktur percakapan standar **Qwen ChatML**:
```
<|im_start|>system
{instruction}<|im_end|>
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
```

#### Kode Sel:
```python
# Sel 5: Load JSONL dataset
from datasets import load_dataset

dataset_path = "data/credit_finetune_dataset.jsonl"
dataset = load_dataset("json", data_files=dataset_path, split="train")

# Sel 6: Fungsi format prompt ChatML
prompt_template = """<|im_start|>system
{}<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
{}<|im_end|>"""

EOS_TOKEN = tokenizer.eos_token if tokenizer.eos_token else "<|im_end|>"

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = prompt_template.format(instruction, input_text, output)
        texts.append(text)
    return {"text": texts}

formatted_dataset = dataset.map(formatting_prompts_func, batched=True)
print("Contoh formatted prompt:\n", formatted_dataset[0]["text"][:500])
```

---

### Bagian 5: Konfigurasi SFTTrainer (TRL) & Eksekusi Training

#### Tujuan:
Melatih model selama 2–3 epoch menggunakan `trl.SFTTrainer` dengan optimizer 8-bit AdamW dan learning rate scheduler linear/cosine.

#### Rekomendasi Parameter Training:
| Parameter | Nilai | Alasan |
| :--- | :--- | :--- |
| `per_device_train_batch_size` | `2` | Mencegah CUDA OOM di GPU T4 16GB atau RTX 8GB |
| `gradient_accumulation_steps`| `4` | Efektif batch size = $2 \times 4 = 8$ |
| `learning_rate` | `2e-4` | Nilai standar optimal untuk QLoRA rank 16 |
| `num_train_epochs` | `2` atau `3` | Konvergensi loss yang stabil tanpa overfit |
| `optim` | `"paged_adamw_8bit"` | Menghemat alokasi VRAM optimizer dan stabil |
| `weight_decay` | `0.01` | Regularisasi bobot adapter |
| `lr_scheduler_type` | `"cosine"` atau `"linear"` | Penurunan learning rate bertahap |

#### Kode Sel:
```python
# Sel 7: Setup SFTTrainer
from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="outputs/credit_qwen_qlora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    save_strategy="epoch",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=training_args,
)

# Sel 8: Eksekusi Training & Monitor Loss
trainer_stats = trainer.train()
print(f"✅ Training selesai! Final Loss: {trainer_stats.training_loss:.4f}")
```

---

### Bagian 6: Evaluasi Inferensi & Validasi Skema JSON

#### Tujuan:
Menguji model yang telah di-fine-tune untuk memvalidasi penalaran underwriting dan kepatuhan output terhadap skema JSON memo.

#### Kode Sel:
```python
# Sel 9: Setup inference mode & uji coba output JSON
import json

model.eval()

sample_instruction = "Anda adalah Senior Credit Risk Underwriter AI di institusi perbankan. Tugas Anda adalah mengevaluasi aplikasi kredit pemohon berdasarkan data demografi, keuangan, hasil prediksi model XGBoost (Probability of Default), dan kontribusi faktor risiko matematis (SHAP Values).\n\nHasilkan laporan analisis kredit (Credit Underwriting Memo) yang terstruktur strictly dalam format JSON valid."

sample_input = """### PROFIL PEMOHON PINJAMAN:
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
  * Pendapatan tahunan sebesar $45,000 (SHAP: -0.35)"""

inference_prompt = prompt_template.format(sample_instruction, sample_input, "")
inputs = tokenizer([inference_prompt], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )

response_text = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
generated_response = response_text.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()

print("Hasil Respon Model:\n", generated_response)

# Validasi JSON Parser
try:
    parsed_json = json.loads(generated_response)
    print("\n✅ JSON VALID! Rekomendasi:", parsed_json.get("recommendation"))
except json.JSONDecodeError as e:
    print("\n❌ JSON Parsing Gagal:", str(e))
```

---

### Bagian 7: Simpan LoRA Adapter & Merge Bobot Model

#### Tujuan:
1. Menyimpan LoRA Adapter (`adapter_model.bin` / `safetensors`).
2. Melakukan *merge and unload* LoRA ke bobot base model untuk diekspor ke format GGUF bagi Ollama.

#### Kode Sel:
```python
# Sel 10: Simpan LoRA Adapter
adapter_dir = "models/lora_adapter"
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print(f"✅ LoRA Adapter tersimpan di: {adapter_dir}")

# Sel 11: Merge LoRA Adapter ke Base Model (Float16) untuk Export GGUF
from peft import PeftModel

base_model_reload = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

merged_model = PeftModel.from_pretrained(base_model_reload, adapter_dir)
merged_model = merged_model.merge_and_unload()

merged_dir = "models/merged_model"
merged_model.save_pretrained(merged_dir)
tokenizer.save_pretrained(merged_dir)
print(f"✅ Merged Model tersimpan di: {merged_dir}")
```

---

### Bagian 8: Membuat `Modelfile` untuk Ollama Serving

#### Tujuan:
Membuat file konfigurasi deklaratif `models/Modelfile` untuk Ollama serving.

#### Kode Sel:
```python
# Sel 12: Buat Modelfile untuk Ollama
modelfile_content = """FROM ./qwen2.5-credit-risk-q4_k_m.gguf

TEMPLATE \"\"\"{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ end }}\"\"\"

SYSTEM \"\"\"Anda adalah Senior Credit Risk Underwriter AI di institusi perbankan. Tugas Anda adalah mengevaluasi aplikasi kredit pemohon berdasarkan data demografi, keuangan, hasil prediksi model XGBoost (Probability of Default), dan kontribusi faktor risiko matematis (SHAP Values). Hasilkan laporan analisis kredit (Credit Underwriting Memo) yang terstruktur strictly dalam format JSON valid.\"\"\"

PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
"""

with open("models/Modelfile", "w", encoding="utf-8") as f:
    f.write(modelfile_content)

print("✅ Modelfile berhasil dibuat di models/Modelfile")
```

---

## 📋 Checklist Eksekusi Mandiri

- [ ] Dependensi `transformers`, `peft`, `bitsandbytes`, `accelerate`, `trl`, `datasets` terpasang.
- [ ] GPU terdeteksi dan Base model `Qwen/Qwen2.5-7B-Instruct` dimuat dalam 4-bit NF4.
- [ ] Dataset `data/credit_finetune_dataset.jsonl` (1.200 baris) berhasil di-load.
- [ ] Prompt diformat sesuai tag ChatML (`<|im_start|>system...`, `<|im_start|>user...`, `<|im_start|>assistant...`).
- [ ] Training loss mengalami konvergensi dengan `trl.SFTTrainer`.
- [ ] Hasil inferensi uji menghasilkan JSON valid yang lulus `json.loads()`.
- [ ] Adapter LoRA tersimpan di `models/lora_adapter` dan file `models/Modelfile` siap digunakan.
