# 🚀 Dokumen 04: Blueprint & Panduan Fine-Tuning Qwen 2.5 dengan Hugging Face PEFT + TRL (QLoRA 4-bit NF4)

* **Repository:** [https://github.com/difadlyaulhaq/xai-credit-agent](https://github.com/difadlyaulhaq/xai-credit-agent)
* **Notebook Rujukan:** `notebooks/04_peft_trl_finetuning.ipynb` (dan `notebooks/04_unsloth_finetuning.ipynb`)
* **Hardware Target:** Local GPU (NVIDIA GTX 1650 4GB / RTX series) atau Google Colab T4 (15-16 GB VRAM)
* **Base Model:** `Qwen/Qwen2.5-3B-Instruct` (dapat dikonfigurasi ke varian 7B `Qwen/Qwen2.5-7B-Instruct`)
* **Input Dataset:** `data/credit_finetune_dataset.jsonl` (1.200 sampel ChatML)
* **Output:**
  - `models/lora_adapter/` (Bobot LoRA Adapter 57.16 MB & Tokenizer)
  - `models/merged_model/` (Merged FP16 Model untuk konversi GGUF / Ollama)

---

## 📌 Alur Pipeline Fine-Tuning (QLoRA PEFT + TRL)

```
                                      FINE-TUNING WORKFLOW
                                      
  [data/credit_finetune_dataset.jsonl]
                   │
                   ▼
  [1. Environment Setup] ─────────────► PyTorch (CUDA), transformers, peft, trl, bitsandbytes, datasets
                   │
                   ▼
  [2. Load Base Model (4-bit NF4)] ───► AutoModelForCausalLM + BitsAndBytesConfig (device_map={"": 0})
                   │
                   ▼
  [3. Attach QLoRA Adapters] ─────────► LoraConfig (r=8, alpha=16, target_modules=All Linear Projections)
                   │
                   ▼
  [4. ChatML Formatting & Tokenize] ──► <|im_start|>system...<|im_start|>user...<|im_start|>assistant...
                   │
                   ▼
  [5. SFT Training Execution] ────────► TRL SFTConfig & SFTTrainer (Batch Size 1, Grad Accum 4, AdamW 8-bit)
                   │
                   ▼
  [6. Validation & JSON Parsing Test] ► Evaluasi Inferensi & Validasi Skema JSON Memo (json.loads)
                   │
                   ▼
  [7. Save Adapter & Merging FP16] ──► Save to models/lora_adapter (57.1 MB) & merge ke CPU RAM
```

---

## 🛠️ Rincian Implementasi Teknis & Solusi Masalah

### 1. Kuantisasi 4-bit NF4 & Device Map Fix pada VRAM 4GB
Pada GPU dengan VRAM terbatas (4GB seperti GTX 1650), penggunaan `device_map="auto"` dapat menyebabkan `accelerate` memecah (*dispatch*) layer ke CPU sehingga memicu `ValueError` pada `bitsandbytes`.
**Solusi**:
```python
device_map = {"": 0} if torch.cuda.is_available() else None
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
)
```

### 2. Konfigurasi LoRA Hyperparameters (PEFT)
* **Rank (`r`)**: `8`
* **LoRA Alpha (`lora_alpha`)**: `16` (Scaling ratio $lpha/r = 2$)
* **Target Modules**: Seluruh proyeksi linear attention dan MLP (`["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`)
* **LoRA Dropout**: `0.05`
* **Task Type**: `CAUSAL_LM`

### 3. Konfigurasi Training TRL Modern (`SFTConfig` & `SFTTrainer`)
Pada pustaka TRL v0.24+, argumen SFT dikonfigurasi melalui `SFTConfig` dan tokenizer dipassing melalui parameter `processing_class`:
```python
training_args = SFTConfig(
    output_dir="outputs/credit_qwen_qlora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    save_strategy="no",
    report_to="none",
    dataset_text_field="text",
    max_length=384,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    processing_class=tokenizer,
    args=training_args,
)
```

### 4. Hasil & Metrik Konvergensi Loss
* **Step 10**: Loss = `1.4120`
* **Step 30**: Loss = `0.1708`
* **Step 50**: Loss = `0.1449`
* **Step 90**: Loss = `0.1276` (Konvergensi optimal tercapai)

---

## 📋 Checklist Eksekusi
- [x] Dependensi `torch (CUDA)`, `transformers`, `peft`, `bitsandbytes`, `trl`, `datasets` terpasang.
- [x] Base model `Qwen/Qwen2.5-3B-Instruct` dimuat dalam 4-bit NF4 ke GPU.
- [x] Dataset `credit_finetune_dataset.jsonl` diformat sesuai tag ChatML.
- [x] Training konvergen hingga loss mencapai **0.1276**.
- [x] Pengujian inferensi menghasilkan memo analisis kredit dengan format JSON 100% valid.
- [x] LoRA adapter tersimpan di `models/lora_adapter` (57.16 MB) dan siap digunakan di aplikasi.
