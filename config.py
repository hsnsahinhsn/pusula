"""
Türkçe Karar Modeli (Laya) & Sentetik Veri Pipeline Konfigürasyonu
NVIDIA RTX 4090 (24 GB VRAM) donanımına optimize edilmiştir.
"""
from dataclasses import dataclass, field
from typing import List, Dict
import os

@dataclass
class GenerationConfig:
    # Yerel LLM Sunucusu (vLLM veya Ollama)
    api_base: str = os.getenv("LLM_API_BASE", "http://localhost:8000/v1")
    api_key: str = os.getenv("LLM_API_KEY", "EMPTY")
    model_name: str = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 450
    request_timeout: int = 60
    
    # Üretim Hedefleri (Kategori başına örnek sayısı)
    samples_per_category: int = 1500  # Toplam 6000 sentetik veri hedefi
    batch_size: int = 16
    output_dir: str = "data"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # Dikey Senaryolar ve Taksonomiler
    taxonomies: Dict[str, Dict] = field(default_factory=lambda: {
        "tool_calling": {
            "description": "Ajanın kullanıcı niyetine göre çağıracağı en uygun aracı seçmesi.",
            "options": [
                "web_search",          # İnternette güncel bilgi arama
                "sql_query",           # Veritabanı sorgulama
                "crm_lookup",          # Müşteri profili / sipariş getirme
                "calculator",          # Sayısal ve finansal hesaplama
                "calendar_booking",    # Randevu ve takvim organizasyonu
                "send_email",          # E-posta oluşturma ve gönderme
                "code_interpreter",    # Kod çalıştırma ve analiz
                "human_handoff"        # Temsilciye devir
            ]
        },
        "customer_service": {
            "description": "Gelen müşteri talep/şikayet metnini doğru departmana yönlendirme.",
            "options": [
                "muhasebe_finans",     # Fatura, iade, ödeme, borç
                "teknik_destek",       # Hata, sistem arızası, bağlantı sorunu
                "satis_pazarlama",     # Ürün teklifi, yeni abonelik, demo talebi
                "lojistik_kargo",      # Sevkiyat, kargo takibi, teslimat gecikmesi
                "hukuk_kvkk",          # Veri silme, sözleşme, yasal ihtarlar
                "insan_kaynaklari",    # İş başvurusu, staj, mülakat
                "genel_destek"         # Basit bilgi talepleri, mesai saatleri
            ]
        },
        "security_guardrails": {
            "description": "Kullanıcı girdisinin sistem güvenlik ve uyum politikalarına uygunluk denetimi.",
            "options": [
                "guvenli_istek",           # Temiz ve meşru kullanıcı isteği
                "prompt_injection",        # Sistem talimatlarını ezmeye çalışan saldırılar
                "jailbreak_attempt",       # Modelin ahlaki ve güvenlik kurallarını delme çabası
                "pii_veri_sizintisi",      # TC Kimlik, Kredi kartı, şifre ifşa riski
                "zararli_icerik",          # Tehdit, küfür, hakaret, nefret söylemi
                "sistem_komut_ihlali"      # Sunucuya bash/SQL komutu çalıştırma girişimi
            ]
        },
        "urgency_scoring": {
            "description": "Talebin aciliyet seviyesi ve SLA kritikliği derecelendirmesi.",
            "options": [
                "kritik_acil",         # Sistem çöktü, para kaybı var, anında müdahale gerek
                "yuksek_oncelik",      # Önemli iş aksaması, 1-2 saat içinde çözüm gerek
                "normal_oncelik",      # Günlük standart iş akışı, gün içi çözüm
                "dusuk_oncelik"        # Zaman kısıtı olmayan genel soru veya geri bildirim
            ]
        }
    })


@dataclass
class TrainingConfig:
    # Model Omurgası
    # Tavsiye 1: convaiinnovations/laya-multilingual (mmBERT-base, 100+ dil)
    # Tavsiye 2: answerdotai/ModernBERT-large (421M params, en hızlı ve güçlü encoder)
    base_model_name: str = "convaiinnovations/laya-multilingual"
    fallback_model_name: str = "answerdotai/ModernBERT-large"
    
    max_length: int = 512
    batch_size: int = 32          # RTX 4090 24GB VRAM için ideal batch
    gradient_accumulation_steps: int = 2
    learning_rate_backbone: float = 2e-5
    learning_rate_heads: float = 1e-4
    weight_decay: float = 0.01
    num_train_epochs: int = 5
    warmup_ratio: float = 0.1
    
    # Multi-task Kayıp Ağırlıkları
    loss_weight_choice: float = 1.0
    loss_weight_score: float = 0.5
    loss_weight_action: float = 0.5
    
    # Donanım & Hızlandırma
    mixed_precision: str = "bf16"  # RTX 4090 bfloat16 donanım desteği
    use_flash_attention: bool = True
    torch_compile: bool = True
    output_dir: str = "checkpoints/laya_turkish_decision"
    save_total_limit: int = 2


@dataclass
class InferenceConfig:
    checkpoint_dir: str = "checkpoints/laya_turkish_decision/best"
    device: str = "cuda"
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    # Kalibrasyon sıcaklık çarpanı
    temperature_scaling: float = 1.0
