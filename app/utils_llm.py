import os
import json
import requests
import torch

ADAPTER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'lora_adapter'))
BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

_local_pipeline = None

def get_local_llm():
    """
    Memuat model PyTorch LoRA Adapter secara lazy untuk inferensi lokal.
    """
    global _local_pipeline
    if _local_pipeline is None and os.path.exists(ADAPTER_DIR):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=True
            )
            device_map = {"": 0} if torch.cuda.is_available() else "cpu"
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                quantization_config=bnb_config if torch.cuda.is_available() else None,
                device_map=device_map,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
            model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
            model.eval()
            _local_pipeline = (model, tokenizer)
        except Exception as e:
            print(f"[WARN] Local LLM loading error: {e}")
            _local_pipeline = None
    return _local_pipeline

def format_underwriting_prompt(raw_input: dict, ml_result: dict) -> tuple[str, str]:
    """
    Menyusun prompt ChatML yang berisi profil pemohon, hasil XGBoost, dan SHAP values.
    """
    system_instruction = (
        "Anda adalah Senior Credit Risk Underwriter AI di institusi perbankan. "
        "Tugas Anda adalah mengevaluasi aplikasi kredit pemohon berdasarkan data demografi, keuangan, "
        "hasil prediksi model XGBoost (Probability of Default), dan kontribusi faktor risiko matematis (SHAP Values).\n\n"
        "Hasilkan laporan analisis kredit (Credit Underwriting Memo) yang terstruktur strictly dalam format JSON valid."
    )

    pd_pct = ml_result['probability_of_default'] * 100.0
    risk_level = ml_result['risk_level']
    rec = ml_result['recommendation']

    # Format drivers
    drivers_str = ""
    for d in ml_result['top_risk_drivers']:
        drivers_str += f"  * {d['feature']} bernilai {d['value']} (SHAP: +{d['shap_value']:.2f})\n"
    if not drivers_str:
        drivers_str = "  * Tidak ada faktor pendorong risiko signifikan.\n"

    # Format mitigating
    mitigating_str = ""
    for m in ml_result['mitigating_factors']:
        mitigating_str += f"  * {m['feature']} bernilai {m['value']} (SHAP: {m['shap_value']:.2f})\n"
    if not mitigating_str:
        mitigating_str = "  * Tidak ada faktor pereda risiko signifikan.\n"

    user_input = f"""### PROFIL PEMOHON PINJAMAN:
- Usia Pemohon: {raw_input.get('person_age')} tahun
- Pendapatan Tahunan: ${raw_input.get('person_income'):,.0f}
- Status Kepemilikan Rumah: {raw_input.get('person_home_ownership')}
- Lama Bekerja: {raw_input.get('person_emp_length')} tahun
- Tujuan Pinjaman: {raw_input.get('loan_intent')}
- Peringkat Risiko Kredit (Grade): {raw_input.get('loan_grade')}
- Besaran Pinjaman yang Diajukan: ${raw_input.get('loan_amnt'):,.0f}
- Suku Bunga Pinjaman: {raw_input.get('loan_int_rate')}%
- Rasio Pinjaman / Pendapatan: {(raw_input.get('loan_amnt') / raw_input.get('person_income') * 100):.1f}%
- Riwayat Gagal Bayar Sebelumnya: {raw_input.get('cb_person_default_on_file')}
- Panjang Riwayat Kredit: {raw_input.get('cb_person_cred_hist_length')} tahun

### KALKULASI RISIKO ML & ANALISIS SHAP:
- Prediksi Probability of Default (PD): {pd_pct:.1f}%
- Kategori Risiko Awal: {risk_level}
- Rekomendasi Awal: {rec}
- Faktor Pendorong Risiko Terbesar (+SHAP):
{drivers_str}- Faktor Pereda Risiko Terbesar (-SHAP):
{mitigating_str}"""

    return system_instruction, user_input

def generate_rule_based_fallback(raw_input: dict, ml_result: dict) -> dict:
    """
    Fallback generator cerdas berbasis aturan domain kredit untuk menjamin ketersediaan 100% 
    jika backend LLM lokal/cloud sedang offline atau kehabisan kuota.
    """
    pd_pct = ml_result['probability_of_default'] * 100.0
    rec = ml_result['recommendation']
    risk = ml_result['risk_level']
    loan_pct = (raw_input.get('loan_amnt', 10000) / raw_input.get('person_income', 50000)) * 100.0

    drivers = [
        f"Rasio pinjaman terhadap pendapatan sebesar {loan_pct:.1f}% (Beban pinjaman relatif terhadap arus kas tahunan)",
        f"Peringkat risiko pinjaman grade {raw_input.get('loan_grade', 'C')} dengan suku bunga {raw_input.get('loan_int_rate', 12)}%"
    ]
    if raw_input.get('cb_person_default_on_file') == 'Y':
        drivers.insert(0, "Terdapat catatan riwayat gagal bayar kredit sebelumnya pada file biro kredit.")

    mitigations = [
        f"Pendapatan tahunan pemohon sebesar ${raw_input.get('person_income', 50000):,.0f} memberikan kapasitas bayar bulanan.",
        f"Masa kerja {raw_input.get('person_emp_length', 2)} tahun dan panjang riwayat kredit {raw_input.get('cb_person_cred_hist_length', 3)} tahun menunjukkan rekam jejak aktif."
    ]

    if rec == 'APPROVE':
        summary = (
            f"Aplikasi pinjaman sebesar ${raw_input.get('loan_amnt', 0):,.0f} direkomendasikan untuk DISETUJUI. "
            f"Probabilitas gagal bayar rendah ({pd_pct:.1f}%) dan faktor pereda risiko mendominasi profil nasabah. "
            "Plafon dan suku bunga standar dapat langsung diterapkan."
        )
        conditions = ["Penandatanganan akad pinjaman standar", "Autodebet cicilan dari rekening penggajian"]
    elif rec == 'MANUAL_REVIEW':
        summary = (
            f"Aplikasi pinjaman sebesar ${raw_input.get('loan_amnt', 0):,.0f} memerlukan REVIEW MANUAL oleh Komite Kredit. "
            f"Probabilitas risiko berada pada rentang moderat ({pd_pct:.1f}%). "
            "Diperlukan verifikasi dokumen pendapatan terkini dan konfirmasi kestabilan tempat tinggal sebelum pencairan."
        )
        conditions = [
            "Verifikasi rekening koran / mutasi bank 3 bulan terakhir",
            "Pertimbangan penyesuaian tenor pinjaman untuk menurunkan rasio cicilan bulanan",
            "Konfirmasi bukti domisili tempat tinggal"
        ]
    else:
        summary = (
            f"Aplikasi pinjaman sebesar ${raw_input.get('loan_amnt', 0):,.0f} direkomendasikan untuk DITOLAK. "
            f"Probabilitas default sangat tinggi ({pd_pct:.1f}%) yang didorong oleh kombinasi rasio beban utang tinggi dan profil risiko kredit pemohon."
        )
        conditions = ["Penolakan aplikasi dengan surat pemberitahuan resmi mengenai rasio kapasitas bayar"]

    return {
        "recommendation": rec,
        "risk_level": risk,
        "probability_of_default": f"{pd_pct:.1f}%",
        "key_risk_drivers": drivers,
        "mitigating_factors": mitigations,
        "conditions_or_mitigation_plan": conditions,
        "executive_summary": summary,
        "source": "Agentic Rule-Grounded Engine (Fallback)"
    }

def generate_credit_memo(raw_input: dict, ml_result: dict, mode: str = "auto", ollama_url: str = "http://localhost:11434") -> dict:
    """
    Menghasilkan Credit Underwriting Memo terstruktur.
    Mode:
    - 'local_adapter': Menggunakan PyTorch LoRA Adapter di models/lora_adapter
    - 'ollama': Menggunakan REST API Ollama lokal
    - 'auto': Coba local adapter / Ollama, jika tidak tersedia fallback ke Rule-Based Engine
    """
    sys_prompt, usr_prompt = format_underwriting_prompt(raw_input, ml_result)

    # 1. Coba Ollama jika diminta / auto
    if mode in ['ollama', 'auto']:
        try:
            resp = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": "qwen2.5:3b",
                    "prompt": f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{usr_prompt}<|im_end|>\n<|im_start|>assistant\n",
                    "stream": False,
                    "options": {"temperature": 0.2, "top_p": 0.9}
                },
                timeout=5
            )
            if resp.status_code == 200:
                text_out = resp.json().get('response', '').strip()
                # Parse JSON
                clean_json = text_out.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(clean_json)
                parsed['source'] = "Ollama Local LLM API (Qwen 2.5)"
                return parsed
        except Exception:
            pass

    # 2. Coba Local PyTorch LoRA Adapter jika ada GPU / torch
    if mode in ['local_adapter', 'auto'] and torch.cuda.is_available():
        llm_pack = get_local_llm()
        if llm_pack is not None:
            try:
                model, tokenizer = llm_pack
                full_prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{usr_prompt}<|im_end|>\n<|im_start|>assistant\n"
                inputs = tokenizer([full_prompt], return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=400,
                        temperature=0.2,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id
                    )
                out_text = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
                resp_text = out_text.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
                clean_json = resp_text.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(clean_json)
                parsed['source'] = "Local Fine-Tuned LoRA Adapter (PyTorch CUDA)"
                return parsed
            except Exception as e:
                print(f"[WARN] Local LoRA generation error: {e}")

    # 3. Fallback cerdas berbasis SHAP & XGBoost
    return generate_rule_based_fallback(raw_input, ml_result)
