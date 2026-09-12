import os
import json
import requests

ADAPTER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'lora_adapter'))
MERGED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'merged_model'))
BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

_local_pipeline = None

FEATURE_NAME_MAP = {
    "person_age": "Usia Pemohon",
    "person_income": "Pendapatan Tahunan",
    "person_emp_length": "Masa Kerja",
    "loan_amnt": "Besaran Pinjaman",
    "loan_int_rate": "Suku Bunga Pinjaman",
    "loan_percent_income": "Rasio Pinjaman terhadap Pendapatan",
    "cb_person_cred_hist_length": "Panjang Riwayat Kredit",
    "total_loan_cost": "Total Beban Pokok & Bunga Pinjaman",
    "disposable_income_est": "Estimasi Sisa Pendapatan Bersih",
    "emp_to_age_ratio": "Rasio Masa Kerja thdp Usia",
    "cred_hist_to_age_ratio": "Rasio Riwayat Kredit thdp Usia",
    "high_risk_flag": "Indikator Ambang Risiko Tinggi",
    "loan_grade_encoded": "Peringkat Risiko Pinjaman (Grade)",
    "cb_person_default_on_file_encoded": "Catatan Riwayat Gagal Bayar",
    "person_home_ownership_OTHER": "Status Rumah: Lainnya",
    "person_home_ownership_OWN": "Status Rumah: Milik Sendiri",
    "person_home_ownership_RENT": "Status Rumah: Sewa (Rent)",
    "loan_intent_EDUCATION": "Tujuan: Biaya Pendidikan",
    "loan_intent_HOMEIMPROVEMENT": "Tujuan: Renovasi Rumah",
    "loan_intent_MEDICAL": "Tujuan: Biaya Medis/Kesehatan",
    "loan_intent_PERSONAL": "Tujuan: Kebutuhan Pribadi",
    "loan_intent_VENTURE": "Tujuan: Modal Usaha/Bisnis",
}

def check_ollama_availability(ollama_url: str = "http://localhost:11434", target_model: str = "xai-credit-agent") -> tuple[bool, str]:
    """
    Memeriksa ketersediaan server Ollama dan model terkait secara non-blocking (timeout 1.2 detik).
    """
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=1.2)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            # Prioritaskan xai-credit-agent jika ada
            for m in models:
                if "xai-credit-agent" in m.lower():
                    return True, m
            for m in models:
                if target_model in m or "qwen" in m.lower():
                    return True, m
            return False, "MODEL_NOT_FOUND"
    except Exception:
        pass
    return False, "SERVER_OFFLINE"

def discover_available_engines(ollama_url: str = "http://localhost:11434") -> dict:
    """
    Mendeteksi secara dinamis backend AI/LLM yang benar-benar aktif dan tersedia di sistem:
    1. Fast Grounded XAI Rule Engine (berbasis inferensi SHAP TreeExplainer & XGBoost).
    2. Ollama Local REST API (memeriksa model yang terpasang secara real-time via /api/tags).
    3. PyTorch LoRA / Merged Model (memeriksa keberadaan file model di direktori models/).
    """
    engines = {}

    # 1. Mode Otomatis (Rekomendasi Cerdas Dinamis)
    engines["auto"] = {
        "label": "⚡ Otomatis (Rekomendasi Cerdas)",
        "badge": "🚀 Auto-Detect",
        "is_ready": True,
        "detail": "Memilih mesin terbaik yang aktif secara instan & tanpa freeze.",
    }

    # 2. Fast Grounded XAI Rule Engine (Selalu siap, 100% reliabel & instan)
    engines["fast_agent"] = {
        "label": "⚡ Fast SHAP-Grounded Engine",
        "badge": "✅ Siap (<10ms)",
        "is_ready": True,
        "detail": "Inferensi berbasis kontribusi matematis SHAP & aturan underwriting komite kredit.",
    }

    # 3. Deteksi Dinamis Service Ollama & Model Terpasang
    ollama_online = False
    installed_models = []
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=1.0)
        if resp.status_code == 200:
            ollama_online = True
            models_data = resp.json().get("models", [])
            installed_models = [m.get("name") for m in models_data if m.get("name")]
    except Exception:
        ollama_online = False

    if ollama_online:
        if installed_models:
            # Daftarkan setiap model yang benar-benar ada di Ollama secara dinamis
            for m_name in installed_models:
                engines[f"ollama:{m_name}"] = {
                    "label": f"🦙 Ollama: {m_name}",
                    "badge": "✅ Siap (Local GGUF)",
                    "is_ready": True,
                    "model_name": m_name,
                    "detail": f"Model {m_name} aktif di Ollama (port 11434).",
                }
        else:
            engines["ollama:qwen2.5:3b"] = {
                "label": "🦙 Ollama API (qwen2.5:3b)",
                "badge": "⚠️ Belum Diunduh",
                "is_ready": False,
                "model_name": "qwen2.5:3b",
                "detail": "Ollama aktif, tetapi jalankan 'ollama run qwen2.5:3b' untuk mengunduh model.",
            }
    else:
        engines["ollama:offline"] = {
            "label": "🦙 Ollama API",
            "badge": "🔴 Server Offline",
            "is_ready": False,
            "model_name": "qwen2.5:3b",
            "detail": f"Tidak dapat terhubung ke service Ollama di {ollama_url}.",
        }

    # 4. Deteksi File Model PyTorch Lokal di models/
    has_adapter = os.path.exists(ADAPTER_DIR)
    has_merged = os.path.exists(MERGED_DIR)
    has_weights = has_adapter or has_merged

    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception:
        pass

    if has_weights:
        loc = "models/lora_adapter" if has_adapter else "models/merged_model"
        badge = "✅ Siap (GPU CUDA)" if cuda_available else "⚠️ CPU Mode"
        engines["local_adapter"] = {
            "label": f"🔥 PyTorch LoRA ({loc})",
            "badge": badge,
            "is_ready": True,
            "detail": f"Bobot model ditemukan di {loc}. Menggunakan komputasi PyTorch.",
        }
    else:
        engines["local_adapter"] = {
            "label": "🔥 PyTorch LoRA (models/)",
            "badge": "🔴 File Model Tidak Ditemukan",
            "is_ready": False,
            "detail": "Folder model lokal tidak ditemukan di models/.",
        }

    return engines

def get_local_llm():
    """
    Memuat model PyTorch LoRA Adapter secara lazy untuk inferensi lokal.
    Hanya dipanggil jika pengguna secara eksplisit memilih engine PyTorch LoRA.
    """
    global _local_pipeline
    if _local_pipeline is None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel

            # Prioritaskan direktori model lokal yang sudah dimerge jika ada
            source_dir = ADAPTER_DIR if os.path.exists(ADAPTER_DIR) else MERGED_DIR
            tokenizer = AutoTokenizer.from_pretrained(source_dir, trust_remote_code=True)

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
            if os.path.exists(ADAPTER_DIR):
                model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
            else:
                model = base_model
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
        "Hasilkan laporan analisis kredit (Credit Underwriting Memo) strictly dalam format JSON valid dengan struktur persis:\n"
        "{\n"
        '  "recommendation": "APPROVE" | "MANUAL_REVIEW" | "REJECT",\n'
        '  "risk_tier": "LOW_RISK" | "MEDIUM_RISK" | "HIGH_RISK",\n'
        '  "probability_of_default": "<nilai_persen>%",\n'
        '  "key_risk_drivers": ["poin pendorong risiko 1", "poin pendorong risiko 2"],\n'
        '  "mitigating_factors": ["poin pereda risiko 1", "poin pereda risiko 2"],\n'
        '  "special_conditions": ["syarat mitigasi atau pencairan pinjaman"],\n'
        '  "underwriter_memo": "Ringkasan eksekutif naratif pertimbangan kredit dari sudut pandang underwriter perbankan senior."\n'
        "}\n"
        "Pastikan tidak menambahkan teks pembuka atau penutup di luar blok JSON."
    )

    pd_pct = ml_result['probability_of_default'] * 100.0
    risk_level = ml_result['risk_level']
    rec = ml_result['recommendation']

    # Format drivers
    drivers_str = ""
    for d in ml_result.get('top_risk_drivers', []):
        feat_name = FEATURE_NAME_MAP.get(d['feature'], d['feature'])
        drivers_str += f"  * {feat_name} bernilai {d['value']} (SHAP: +{d['shap_value']:.2f})\n"
    if not drivers_str:
        drivers_str = "  * Tidak ada faktor pendorong risiko signifikan.\n"

    # Format mitigating
    mitigating_str = ""
    for m in ml_result.get('mitigating_factors', []):
        feat_name = FEATURE_NAME_MAP.get(m['feature'], m['feature'])
        mitigating_str += f"  * {feat_name} bernilai {m['value']} (SHAP: {m['shap_value']:.2f})\n"
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
- Rasio Pinjaman / Pendapatan: {(raw_input.get('loan_amnt', 10000) / max(raw_input.get('person_income', 50000), 1) * 100):.1f}%
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
    Generator memo cerdas instan (<10ms) berbasis hasil inferensi XGBoost dan kontribusi SHAP TreeExplainer.
    Menghasilkan memo underwriting profesional tanpa ketergantungan model berat.
    """
    pd_pct = ml_result['probability_of_default'] * 100.0
    rec = ml_result['recommendation']
    risk = ml_result['risk_level']
    income = raw_input.get('person_income', 50000)
    loan_amnt = raw_input.get('loan_amnt', 10000)
    loan_pct = (loan_amnt / max(income, 1)) * 100.0
    emp_len = raw_input.get('person_emp_length', 2.0)
    cred_hist = raw_input.get('cb_person_cred_hist_length', 3)
    grade = raw_input.get('loan_grade', 'B')
    rate = raw_input.get('loan_int_rate', 11.0)
    has_default = raw_input.get('cb_person_default_on_file') == 'Y'

    # 1. Konstruksi Key Risk Drivers dari data SHAP aktual
    drivers = []
    top_drivers = ml_result.get('top_risk_drivers', [])
    if top_drivers:
        for d in top_drivers:
            fname = FEATURE_NAME_MAP.get(d['feature'], d['feature'])
            val = d['value']
            shap_val = d['shap_value']
            if "loan_percent" in d['feature'] or "loan_amnt" in d['feature']:
                drivers.append(f"{fname} ({loan_pct:.1f}%) berkontribusi meningkatkan estimasi risiko (SHAP: +{shap_val:.2f})")
            elif "default" in d['feature']:
                drivers.append(f"Catatan riwayat gagal bayar sebelumnya pada biro kredit memperberat skor risiko (SHAP: +{shap_val:.2f})")
            elif "grade" in d['feature'] or "rate" in d['feature']:
                drivers.append(f"{fname} di Grade {grade} ({rate}%) mencerminkan premi risiko tinggi (SHAP: +{shap_val:.2f})")
            else:
                drivers.append(f"{fname} bernilai {val} menjadi faktor pendorong risiko default (SHAP: +{shap_val:.2f})")
    else:
        drivers.append(f"Rasio beban pinjaman terhadap pendapatan sebesar {loan_pct:.1f}%")

    # 2. Konstruksi Mitigating Factors dari data SHAP aktual
    mitigations = []
    top_mitigs = ml_result.get('mitigating_factors', [])
    if top_mitigs:
        for m in top_mitigs:
            fname = FEATURE_NAME_MAP.get(m['feature'], m['feature'])
            shap_val = abs(m['shap_value'])
            if "income" in m['feature'] and "percent" not in m['feature']:
                mitigations.append(f"Kapasitas pendapatan tahunan (${income:,.0f}) menjadi peredam utama kemungkinan default (SHAP: -{shap_val:.2f})")
            elif "emp_length" in m['feature']:
                mitigations.append(f"Stabilitas masa kerja selama {emp_len} tahun memberikan kepastian kontinuitas penghasilan (SHAP: -{shap_val:.2f})")
            elif "cred_hist" in m['feature']:
                mitigations.append(f"Rekam jejak riwayat kredit selama {cred_hist} tahun menunjukkan kebiasaan pengelolaan fasilitas (SHAP: -{shap_val:.2f})")
            else:
                mitigations.append(f"{fname} bernilai positif dalam meredam probabilitas gagal bayar (SHAP: -{shap_val:.2f})")
    else:
        mitigations.append(f"Pendapatan tahunan pemohon sebesar ${income:,.0f} menopang arus kas pembayaran.")

    # 3. Rekomendasi & Syarat
    if rec == 'APPROVE':
        summary = (
            f"Aplikasi pinjaman sebesar ${loan_amnt:,.0f} direkomendasikan untuk DISETUJUI. "
            f"Model XGBoost memprediksi Probability of Default (PD) rendah yaitu {pd_pct:.1f}%. "
            f"Kapasitas pendapatan dan stabilitas kerja pemohon efektif menekan potensi gagal bayar. "
            "Fasilitas kredit dapat dicairkan sesuai struktur plafon yang diajukan."
        )
        conditions = [
            "Penandatanganan perjanjian kredit standar dan persetujuan klausul autodebet rekening",
            "Konfirmasi dokumen identitas resmi pemohon dan slip gaji periode terkini"
        ]
    elif rec == 'MANUAL_REVIEW':
        summary = (
            f"Aplikasi pinjaman sebesar ${loan_amnt:,.0f} diklasifikasikan ke zona moderat (PD: {pd_pct:.1f}%) "
            "dan memerlukan TINJAUAN MANUAL (Manual Review) oleh Komite Kredit. "
            f"Rasio pinjaman terhadap pendapatan berada di angka {loan_pct:.1f}% dengan grade {grade}. "
            "Diperlukan mitigasi protektif sebelum komitmen kredit disetujui."
        )
        conditions = [
            "Verifikasi mutasi rekening koran operasional 3 bulan terakhir untuk validasi arus kas masuk",
            "Opsi penyesuaian tenor pinjaman lebih panjang atau penyesuaian plafon guna menekan rasio angsuran",
            "Penyertaan dokumen jaminan tambahan atau penjamin (guarantor) bila dipandang perlu"
        ]
    else:
        summary = (
            f"Aplikasi pinjaman sebesar ${loan_amnt:,.0f} direkomendasikan untuk DITOLAK (Reject). "
            f"Probabilitas gagal bayar terpantau tinggi ({pd_pct:.1f}%) melebihi ambang batas toleransi risiko institusi. "
            "Kombinasi rasio eksposur utang dan riwayat profil kredit tidak memenuhi parameter underwriting minimum."
        )
        conditions = [
            "Penerbitan surat penolakan resmi (Adverse Action Notice) dengan penjelasan alasan keterbatasan kapasitas bayar",
            "Peluang pengajuan kembali setelah masa observasi minimum 6 bulan dengan perbaikan riwayat kredit"
        ]

    return {
        "recommendation": rec,
        "risk_level": risk,
        "probability_of_default": f"{pd_pct:.1f}%",
        "key_risk_drivers": drivers,
        "mitigating_factors": mitigations,
        "conditions_or_mitigation_plan": conditions,
        "executive_summary": summary,
        "source": "Agentic Rule-Grounded Engine (Instant & Grounded)"
    }

def generate_credit_memo(raw_input: dict, ml_result: dict, mode: str = "auto", ollama_url: str = "http://localhost:11434") -> dict:
    """
    Menghasilkan Credit Underwriting Memo terstruktur.
    Mode:
    - 'fast_agent': Menggunakan Rule-Based SHAP Grounded Engine instan (sangat cepat, 0.01 detik).
    - 'auto': Cek Ollama jika model terpasang. Jika tidak tersedia, langsung gunakan Fast Grounded Engine tanpa freezing.
    - 'ollama': Menggunakan REST API Ollama lokal (memerlukan model 'qwen2.5:3b' atau sejenisnya di Ollama).
    - 'local_adapter': Menggunakan PyTorch LoRA Adapter lokal (membutuhkan waktu komputasi GPU).
    """
    # 1. Mode Fast Agent langsung
    if mode == 'fast_agent':
        return generate_rule_based_fallback(raw_input, ml_result)

    sys_prompt, usr_prompt = format_underwriting_prompt(raw_input, ml_result)

    # 2. Coba Ollama jika mode 'ollama*' atau 'auto'
    if mode.startswith('ollama') or mode == 'auto':
        target_model = mode.split(":", 1)[1] if ":" in mode and not mode.endswith(":offline") else "xai-credit-agent"
        is_ready, model_or_msg = check_ollama_availability(ollama_url, target_model)
        if is_ready:
            try:
                resp = requests.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": model_or_msg,
                        "prompt": f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{usr_prompt}<|im_end|>\n<|im_start|>assistant\n",
                        "stream": False,
                        "format": "json",
                        "options": {"temperature": 0.2, "top_p": 0.9}
                    },
                    timeout=45
                )
                if resp.status_code == 200:
                    text_out = resp.json().get('response', '').strip()
                    clean_json = text_out.replace("```json", "").replace("```", "").strip()
                    parsed = json.loads(clean_json)
                    if isinstance(parsed, dict):
                        if 'underwriter_memo' in parsed and not parsed.get('executive_summary'):
                            parsed['executive_summary'] = parsed['underwriter_memo']
                        elif 'executive_summary' in parsed and not parsed.get('underwriter_memo'):
                            parsed['underwriter_memo'] = parsed['executive_summary']

                        if 'special_conditions' in parsed and not parsed.get('conditions_or_mitigation_plan'):
                            parsed['conditions_or_mitigation_plan'] = parsed['special_conditions']
                        elif 'conditions_or_mitigation_plan' in parsed and not parsed.get('special_conditions'):
                            parsed['special_conditions'] = parsed['conditions_or_mitigation_plan']

                        if 'risk_tier' in parsed and not parsed.get('risk_level'):
                            parsed['risk_level'] = parsed['risk_tier']
                        elif 'risk_level' in parsed and not parsed.get('risk_tier'):
                            parsed['risk_tier'] = parsed['risk_level']

                        parsed['source'] = f"Ollama Local LLM API ({model_or_msg})"
                        return parsed
            except Exception as exc:
                print(f"[WARN] Ollama generation error: {exc}")
        elif mode.startswith('ollama'):
            # Jika user meminta Ollama secara spesifik namun model belum ada / server offline
            fallback = generate_rule_based_fallback(raw_input, ml_result)
            fallback['source'] = f"Ollama Not Ready ({model_or_msg}) -> Grounded Fallback"
            return fallback

    # 3. Coba Local PyTorch LoRA Adapter HANYA jika diminta secara eksplisit
    # (Jangan jalankan di mode 'auto' karena di laptop 4GB VRAM memakan waktu menit dan membekukan UI)
    if mode == 'local_adapter':
        try:
            import torch
            if not torch.cuda.is_available():
                fallback = generate_rule_based_fallback(raw_input, ml_result)
                fallback['source'] = "PyTorch CUDA Tidak Tersedia -> Fallback Engine"
                return fallback
            llm_pack = get_local_llm()
            if llm_pack is not None:
                model, tokenizer = llm_pack
                full_prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{usr_prompt}<|im_end|>\n<|im_start|>assistant\n"
                inputs = tokenizer([full_prompt], return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=350,
                        temperature=0.2,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id
                    )
                out_text = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
                resp_text = out_text.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
                clean_json = resp_text.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(clean_json)
                if isinstance(parsed, dict):
                    if 'underwriter_memo' in parsed and not parsed.get('executive_summary'):
                        parsed['executive_summary'] = parsed['underwriter_memo']
                    elif 'executive_summary' in parsed and not parsed.get('underwriter_memo'):
                        parsed['underwriter_memo'] = parsed['executive_summary']

                    if 'special_conditions' in parsed and not parsed.get('conditions_or_mitigation_plan'):
                        parsed['conditions_or_mitigation_plan'] = parsed['special_conditions']
                    elif 'conditions_or_mitigation_plan' in parsed and not parsed.get('special_conditions'):
                        parsed['special_conditions'] = parsed['conditions_or_mitigation_plan']

                    if 'risk_tier' in parsed and not parsed.get('risk_level'):
                        parsed['risk_level'] = parsed['risk_tier']
                    elif 'risk_level' in parsed and not parsed.get('risk_tier'):
                        parsed['risk_tier'] = parsed['risk_level']

                parsed['source'] = "Local Fine-Tuned LoRA Adapter (PyTorch CUDA)"
                return parsed
        except Exception as e:
            print(f"[WARN] Local LoRA generation error: {e}")

    # 4. Fallback instan berbasis SHAP & XGBoost
    return generate_rule_based_fallback(raw_input, ml_result)
