"""
Türkçe Karar Modeli (Laya) & Çok Sektörlü Sentetik Veri Pipeline Konfigürasyonu
E-Ticaret, Hukuk, Bankacılık, Yazılım, Eğitim, Sınavlar, Ekonomi, Siyaset ve Temel Bilimleri kapsar.
"""
from dataclasses import dataclass, field
from typing import List, Dict
import os

@dataclass
class GenerationConfig:
    # Yerel LLM Sunucusu (vLLM veya Ollama)
    api_base: str = os.getenv("LLM_API_BASE", "http://localhost:8000/v1")
    api_key: str = os.getenv("LLM_API_KEY", "EMPTY")
    # RTX PRO 6000 (96 GB) için Qwen 3.8 27B / 32B; RTX 4090 için Qwen 3.5 9B / 7B
    model_name: str = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen3.5-9B-Instruct")
    temperature: float = 0.75
    top_p: float = 0.9
    max_tokens: int = 500
    request_timeout: int = 90
    
    # Üretim Hedefleri (Kategori başına örnek sayısı)
    samples_per_category: int = 800   # 9 dikey alan x 800 = ~7.200 sentetik veri
    batch_size: int = 16
    output_dir: str = "data"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # Çok Sektörlü Zengin Dikey Senaryolar ve Taksonomiler
    taxonomies: Dict[str, Dict] = field(default_factory=lambda: {
        # 1. ÜST DÜZEY SEKTÖREL YÖNLENDİRME (Global Router)
        "global_sector_routing": {
            "description": "Gelen kullanıcı sorgusunun ait olduğu ana uzmanlık alanını belirleme.",
            "options": [
                "e_ticaret",               # Alışveriş, sipariş, kargo, ürünler
                "hukuk_mevzuat",           # Mahkeme, kanun, sözleşme, haklar
                "bankacilik_finans",       # Para transferi, kredi, hesap, yatırım
                "yazilim_ve_bt",           # Kodlama, sistem mimarisi, hata ayıklama
                "egitim_ve_sinavlar",      # Okul, dersler, YKS, KPSS, ALES hazırlık
                "temel_bilimler",          # Fizik, kimya, biyoloji, matematik teorisi
                "ekonomi_ve_piyasalar",    # Makroekonomi, enflasyon, borsa, döviz
                "siyaset_ve_kamu",         # Politika, kamu yönetimi, diplomasi
                "sistem_guvenligi"         # Güvenlik açığı, kural ihlali, saldırı
            ]
        },
        # 2. HUKUK & YASAL SÜREÇLER
        "law_and_judiciary": {
            "description": "Hukuki talebin alt uzmanlık ve işlem türünü belirleme.",
            "options": [
                "ictihat_ve_emsal_karar",  # Yargıtay/Danıştay emsal arama
                "sozlesme_inceleme",       # Ticari/iş sözleşmesi risk analizi
                "ceza_hukuku",             # Suç duyurusu, soruşturma, şikayet
                "is_ve_tazminat_hukuku",   # Kıdem, ihbar, işe iade davaları
                "kvkk_ve_veri_koruma",     # Veri silme, aydınlatma metni
                "dilekce_ve_ihtarname",    # Resmi hukuki metin taslağı hazırlama
                "uzman_avukat_yonlendirme" # Karmaşık davalarda insan temsilciye devir
            ]
        },
        # 3. BANKACILIK, FİNANS & EKONOMİ
        "banking_finance_economy": {
            "description": "Finansal ve ekonomik işlem/analiz niyetini tespit etme.",
            "options": [
                "kredi_ve_mevduat_hesabi", # Faiz, kredi kartı limiti, kredi başvurusu
                "supheli_islem_ve_dolandiricilik", # Çalıntı kart, izinsiz transfer (Çok Acil!)
                "yatirim_ve_hisse_senedi", # Borsa, fon, kripto, temettü analizi
                "makroekonomik_gosterge",  # TÜFE, TCMB faiz kararları, cari açık
                "fatura_ve_odeme_islemi",  # Otomatik ödeme talimatı, dekont
                "pos_ve_ticari_bankacilik" # Şirket tahsilatları, pos arızası, çek/senet
            ]
        },
        # 4. YAZILIM, BT & GELİŞTİRİCİ ARAÇLARI
        "software_and_it": {
            "description": "Yazılım geliştirme ve sistem yönetimi görevlerini yönlendirme.",
            "options": [
                "hata_ayiklama_debug",     # Stacktrace, syntax ve runtime hataları
                "kod_yazma_ve_refactor",   # Yeni fonksiyon, temiz kod, optimizasyon
                "veritabani_ve_sql",       # Sorgu optimizasyonu, şema tasarımı, migration
                "devops_ve_bulut",         # Docker, Kubernetes, CI/CD, Linux sunucu
                "api_ve_entegrasyon",      # REST, GraphQL, webhook bağlantıları
                "siber_guvenlik_analizi"   # Kod güvenliği, yetkilendirme açıkları
            ]
        },
        # 5. EĞİTİM & SINAVLARA HAZIRLIK
        "education_and_exams": {
            "description": "Eğitim kademeleri ve ulusal sınavlara hazırlık ihtiyaçlarını yönlendirme.",
            "options": [
                "yks_tyt_ayt_hazirlik",    # Üniversite giriş sınavı soru ve konuları
                "kpss_ve_kamu_sinavlari",  # Genel yetenek/kültür, alan bilgisi
                "lgs_hazirlik",            # Ortaokul ve liseye geçiş sınavı
                "ales_ve_dgs",             # Lisansüstü ve dikey geçiş sayısal/sözel
                "pedagojik_konu_anlatimi", # Adım adım kavram öğretimi
                "soru_cozum_rehberligi",   # Doğrudan soru çözümü ve formül uygulama
                "calisma_plani_ve_koc"     # Zaman yönetimi, haftalık ders programı
            ]
        },
        # 6. TEMEL BİLİMLER (FİZİK, KİMYA, BİYOLOJİ, MATEMATİK)
        "stem_sciences": {
            "description": "Temel doğa ve biçimsel bilimlerin soru/araştırma tipini belirleme.",
            "options": [
                "matematik_ve_geometri",   # Türev, integral, cebir, geometrik ispat
                "klasik_ve_modern_fizik",  # Mekanik, termodinamik, kuantum, optik
                "organik_ve_genel_kimya",  # Tepkimeler, stokiyometri, periyodik tablo
                "hucresel_ve_genel_biyoloji", # Genetik, evrim, anatomi, ekoloji
                "bilimsel_ispat_ve_teori", # Teorem kanıtlama, aksiyomlar
                "sayisal_hesaplama_motoru" # Formül ve numerik hesaplama aracına yönlendirme
            ]
        },
        # 7. E-TİCARET, LOJİSTİK & MÜŞTERİ HİZMETLERİ
        "ecommerce_and_retail": {
            "description": "E-ticaret sipariş, teslimat ve müşteri operasyonlarını yönlendirme.",
            "options": [
                "iade_ve_degisim",         # Ürün iade kodu, cayma hakkı
                "kargo_ve_teslimat_takibi",# Geciken paket, kayıp kargo, kurye
                "odeme_ve_fatura_sorunu",  # Mükerrer çekim, e-fatura talebi
                "urun_bilgisi_ve_stok",    # Beden tablosu, özellik sorgulama
                "kampanya_ve_kupon",       # İndirim kodu çalışmıyor, promosyon
                "canli_destek_temsilcisi"  # Çözülemeyen karmaşık şikayette temsilciye aktarım
            ]
        },
        # 8. SİYASET, KAMU YÖNETİMİ & POLİTİKA
        "politics_and_policy": {
            "description": "Siyasi, kamusal ve uluslararası ilişkiler analizlerini yönlendirme.",
            "options": [
                "kamu_politikasi_analizi", # Şehir planlama, çevre ve kamu reformları
                "secim_ve_siyasi_tarih",   # Siyasi partiler, anayasa tarihi, seçim sistemleri
                "uluslararasi_iliskiler",  # Diplomasi, jeopolitik, dış ticaret antlaşmaları
                "tarafsiz_bilgi_talebi",   # Objektif olgusal siyasi bilgi sunumu
                "hukuk_politikasi_iliskisi" # Yasa teklifleri ve meclis süreçleri
            ]
        },
        # 9. SİSTEM GÜVENLİĞİ & GUARDRAILS
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
        }
    })


@dataclass
class TrainingConfig:
    # Model Omurgası (Türkçe için mmBERT çok dillidir ve morfolojiye hakimdir)
    base_model_name: str = "convaiinnovations/laya-multilingual"
    fallback_model_name: str = "answerdotai/ModernBERT-large"
    
    max_length: int = 512
    batch_size: int = 32          # RTX 4090 / 5090 / PRO 6000 için ideal batch
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
    mixed_precision: str = "bf16"
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
    temperature_scaling: float = 1.0
