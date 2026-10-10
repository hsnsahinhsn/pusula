"""
Evrensel Türkçe Karar Modeli (Universal Turkish Decision Engine) Konfigürasyonu
Ulusal/Uluslararası Sınavlar (YKS, KPSS, SAT, ACT, ESAT, IELTS, TOEFL, IMAT vb.),
Hukuk, Sağlık, Bankacılık, Yazılım, STEM, E-Ticaret, Siyaset ve Mühendislik dahil tüm alanları kapsar.
"""
from dataclasses import dataclass, field
from typing import List, Dict
import os

@dataclass
class GenerationConfig:
    # Yerel LLM Sunucusu (vLLM veya Ollama)
    api_base: str = os.getenv("LLM_API_BASE", "http://localhost:8000/v1")
    api_key: str = os.getenv("LLM_API_KEY", "EMPTY")
    # RTX PRO 6000 (96 GB) için Qwen 3.8 27B / 32B; RTX 4090 için Qwen 3.5 9B
    model_name: str = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen3.5-9B-Instruct")
    temperature: float = 0.75
    top_p: float = 0.9
    max_tokens: int = 550
    request_timeout: int = 120
    
    # Kategori başına örnek hedefi
    samples_per_category: int = 600   # 12 ana alan x 600 = ~7.200 zengin sentetik veri
    batch_size: int = 16
    output_dir: str = "data"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # Evrensel (Universal) Dikey Senaryolar ve Taksonomi Havuzu
    taxonomies: Dict[str, Dict] = field(default_factory=lambda: {
        # 1. EVRENSEL MAKRO YÖNLENDİRME (Global Multi-Domain Router)
        "global_macro_routing": {
            "description": "Gelen kullanıcı girdisinin ait olduğu evrensel makro uzmanlık alanını belirleme.",
            "options": [
                "akademik_ve_sinavlar",     # Ulusal ve uluslararası tüm sınavlar ve akademik süreçler
                "saglik_ve_tip",            # Semptomlar, triyaj, randevu, tahlil, medikal bilgi
                "hukuk_ve_mevzuat",         # Mahkeme, dava, sözleşme, kanun, KVKK
                "bankacilik_ve_finans",     # Para transferi, kredi, borsa, kripto, makroekonomi
                "yazilim_ve_yapay_zeka",    # Kodlama, hata ayıklama, veri tabanı, bulut, ML
                "temel_ve_uygulamali_fen",  # Matematik, fizik, kimya, biyoloji, mühendislik
                "e_ticaret_ve_lojistik",    # Alışveriş, kargo, iade, fatura, pazar yerleri
                "siyaset_kamu_diplomasi",   # Politika, e-devlet, kamu politikası, uluslararası ilişkiler
                "dil_ve_kultur_sanat",      # Yabancı dil öğrenimi, çeviri, edebiyat, içerik üretimi
                "guvenlik_ve_guardrails"    # Sistem ihlali, prompt injection, saldırı engelleme
            ]
        },

        # 2. ULUSAL & ULUSLARARASI TÜM SINAVLAR (Global & National Exams)
        "exams_and_admissions": {
            "description": "Türkiye'deki ulusal ve küresel üniversite/dil/yeterlilik sınavlarının türünü belirleme.",
            "options": [
                "yks_tyt_ayt_ydt",          # Türkiye üniversiteye giriş (Temel/Alan/Yabancı Dil)
                "lgs_ortaogretim",          # Liselere geçiş sınavı
                "kpss_kamu_personeli",      # Genel yetenek/kültür, eğitim bilimleri, ÖABT
                "ales_ve_dgs",              # Lisansüstü eğitim ve dikey geçiş sınavları
                "tus_ve_dus",               # Tıpta ve Diş Hekimliğinde Uzmanlık Sınavı
                "sat_ve_act",               # ABD/Küresel üniversite kabul sınavları (Math, Reading/Writing)
                "esat_ve_step",             # Cambridge/Imperial mühendislik ve doğa bilimleri kabul sınavı
                "imat_ve_bmat",             # İtalya/Avrupa uluslararası tıp fakültesi kabul sınavları
                "ielts_ve_toefl_pte",       # Uluslararası İngilizce dil yeterlilik sınavları
                "gre_ve_gmat",              # Küresel yüksek lisans, MBA ve doktora kabul sınavları
                "ydil_ve_yokdil",           # YDS, e-YDS ve YÖKDİL akademik dil sınavları
                "soru_cozumu_ve_formul",    # Doğrudan soru çözümü, formül ispatı ve çözüm adımları
                "calisma_programi_ve_koc"   # Sınav hazırlık koçluğu, net artırma stratejisi
            ]
        },

        # 3. SAĞLIK, TIP & BİYOMEDİKAL (Health & Triage)
        "health_and_triage": {
            "description": "Sağlık ve tıbbi danışma taleplerinin aciliyet ve branş yönlendirmesi.",
            "options": [
                "acil_vaka_ve_triyaj",      # Kalp krizi, travma, boğulma, 112 acil durum (ÇOK ACİL!)
                "dahiliye_ve_genel_tip",    # Enfeksiyon, tansiyon, halsizlik, diyabet
                "pediyatri_cocuk_sagligi",  # Çocuk hastalıkları, aşı takvimi, ateş
                "ortopedi_ve_fizik_tedavi", # Eklem ağrısı, kırık/çıkık, bel fıtığı
                "psikoloji_ve_ruh_sagligi", # Anksiyete, depresyon, terapi desteği
                "laboratuvar_ve_tahlil",    # Kan/idrar tahlili, MR, Röntgen yorumlama uyarısı
                "ilac_etkilesimi_ve_eczane" # Yan etki riski, doz aşımı kontrolü
            ]
        },

        # 4. HUKUK, ADALET & YASAL SÜREÇLER (Law & Legal)
        "law_and_justice": {
            "description": "Hukuki ihtilafların alt alanını ve aksiyon türünü tespit etme.",
            "options": [
                "yargitay_danistay_ictihat",# Emsal karar ve içtihat tarama
                "sozlesmeler_hukuku",       # Ticari kira, satış, gizlilik sözleşmesi inceleme
                "is_ve_tazminat_hukuku",    # İşe iade, kıdem/ihbar, mobbing
                "ceza_ve_sorusturma",       # Savcılık şikayeti, suç duyurusu, ifade
                "aile_ve_miras_hukuku",     # Boşanma, nafaka, veraset ilamı
                "kvkk_ve_bilisim_hukuku",   # Kişisel veri silme, siber suçlar
                "resmi_dilekce_taslagi",    # Mahkeme veya icra müdürlüğü dilekçesi
                "uzman_avukat_yonlendirme"  # Bağlayıcı yasal tavsiye için vekile aktarım
            ]
        },

        # 5. BANKACILIK, FİNANS, KRİPTO & EKONOMİ (Finance & Economy)
        "banking_finance_crypto": {
            "description": "Finansal işlemler, yatırımlar ve piyasa analizlerini sınıflandırma.",
            "options": [
                "supheli_islem_ve_fraud",   # Çalıntı kart, izinsiz EFT, hesap güvenliği (Acil Eylem!)
                "kredi_faiz_ve_mevduat",    # Konut/ihtiyaç kredisi, vadeli mevduat faizi
                "bist_hisse_ve_fonlar",     # Borsa İstanbul, fon portföyü, temettü
                "kripto_ve_dijital_varlik", # Bitcoin, Ethereum, blockchain transferi
                "makroekonomi_ve_enflasyon",# TÜFE, TCMB politika faizi, döviz kurları
                "fatura_vergi_ve_muhasebe", # Gelir vergisi, KDV iadesi, e-arşiv fatura
                "pos_ve_sirket_tahsilati"   # Ticari bankacılık, bloke, çek/senet
            ]
        },

        # 6. YAZILIM, YAPAY ZEKA & SİBER GÜVENLİK (Software & AI)
        "software_ai_security": {
            "description": "Yazılım mühendisliği, sistem yönetimi ve yapay zeka görevlerini yönlendirme.",
            "options": [
                "hata_ayiklama_debug",      # Runtime error, NullPointer, segfault çözümü
                "kod_yazma_ve_refactor",    # Temiz kod, modüler fonksiyon, tasarım desenleri
                "veritabani_ve_sql",        # PostgreSQL, MySQL, NoSQL indeksleme ve sorgu
                "devops_docker_kubernetes", # CI/CD pipeline, container, Linux sunucu yönetimi
                "yapay_zeka_ve_mlops",      # PyTorch, LLM, model eğitimi, embedding, RAG
                "api_ve_mikroservis",       # FastAPI, Spring, gRPC, webhook entegrasyonu
                "siber_guvenlik_ve_pentest" # OWASP açık tarama, XSS, SQLi, yetkilendirme
            ]
        },

        # 7. TEMEL BİLİMLER (MATEMATİK, FİZİK, KİMYA, BİYOLOJİ) (STEM)
        "stem_sciences_advanced": {
            "description": "Doğa bilimleri ve ileri matematiksel sorguların bilim dalını tespit etme.",
            "options": [
                "ileri_matematik_ve_cebir", # Lineer cebir, çok değişkenli kalkülüs, diferansiyel
                "geometri_ve_topoloji",     # Analitik geometri, vektörler, uzamsal ispatlar
                "klasik_ve_kuantum_fizik",  # Bağıntılar, elektromanyetizma, rölativite, termodinamik
                "organik_ve_fizikokimya",   # Kimyasal kinetik, reaksiyon mekanizmaları, molarite
                "molekuler_biyoloji_genetik",# DNA/RNA, transkripsiyon, hücresel solunum, enzimler
                "sayisal_hesaplama_araci",  # Sembolik matematik motoru (Wolfram/SymPy) çalıştırma
                "bilimsel_makale_ve_teori"  # Akademik literatür tarama ve teorem analizi
            ]
        },

        # 8. MÜHENDİSLİK & İMALAT TEKNOLOJİLERİ (Engineering)
        "applied_engineering": {
            "description": "Mühendislik disiplinleri ve endüstriyel standartları sınıflandırma.",
            "options": [
                "elektrik_elektronik_muh",  # Devre tasarımı, mikrokontrolcüler, gömülü sistemler
                "makine_ve_mekatronik_muh", # Mukavemet, akışkanlar mekaniği, CAD/CAM tasarımı
                "insaat_ve_yapi_muh",       # Statik hesaplama, deprem yönetmeliği, betonarme
                "endustri_ve_verimlilik",   # Yöneylem araştırması, tedarik zinciri optimizasyonu
                "teknik_standartlar_tse_iso" # Kalite kontrol, CE/ISO sertifikasyon şartları
            ]
        },

        # 9. E-TİCARET, PAZAR YERLERİ & MÜŞTERİ HİZMETLERİ (E-Commerce)
        "ecommerce_retail_marketplaces": {
            "description": "E-ticaret, sipariş ve pazar yeri (Trendyol/Amazon vb.) operasyonları.",
            "options": [
                "iade_degisim_ve_cayma",    # İade kargo kodu, ayıplı mal bildirimi
                "kargo_ve_lojistik_gecikme",# Transfer merkezinde takılan paket, kurye şikayeti
                "odeme_ve_tahsilat_uyusmazligi", # Karttan çekilip sipariş oluşmaması
                "urun_stok_ve_uyumluluk",   # Beden tablosu, orijinal ürün teyidi, teknik özellik
                "satici_ve_magaza_yonetimi",# Komisyon oranları, fatura yükleme, stok entegrasyonu
                "canli_insan_temsilcisine_devir" # Botun yetersiz kaldığı durumlarda müşteri temsilcisi
            ]
        },

        # 10. SİYASET, KAMU YÖNETİMİ & ULUSLARARASI İLİŞKİLER (Politics & Policy)
        "politics_public_diplomacy": {
            "description": "Kamu politikası, anayasal süreçler ve küresel diplomasi analizi.",
            "options": [
                "kamu_yonetimi_ve_edevlet",  # Resmi kurum başvuruları, belediye hizmetleri
                "secim_ve_anayasa_surecleri",# Seçim kanunları, meclis denetimi, referandumlar
                "uluslararasi_iliskiler_jeopolitik", # BM, AB, NATO ilişkileri, sınır antlaşmaları
                "tarafsiz_siyasi_bilgi_sunumu", # Yansız, olgusal ve tarafsız perspektif
                "sosyoekonomik_kamu_reformu" # Sosyal güvenlik, eğitim ve sağlık reform analizleri
            ]
        },

        # 11. DİL BİLİM, ÇEVİRİ & KÜLTÜR SANAT (Linguistics & Content)
        "linguistics_and_translation": {
            "description": "Yabancı dil öğrenimi, profesyonel çeviri ve metin analizi.",
            "options": [
                "akademik_ve_yeminli_ceviri",# Resmi belge, makale ve tez çevirisi
                "ingilizce_dilbilgisi_ve_kelime", # Grammar kuralları, phrasal verbs, idioms
                "ikinci_yabanci_dil",       # Almanca, Fransızca, İspanyolca, Rusça, Arapça
                "metin_duzeltme_ve_proofread", # Yazım/noktalama, akıcılık, intihal kontrolü
                "yaratici_metin_ve_senaryo" # Reklam metni, edebi çeviri, yaratıcı yazarlık
            ]
        },

        # 12. GÜVENLİK, SİBER TEHDİTLER & GUARDRAILS (Security & Safety)
        "security_guardrails_universal": {
            "description": "Kullanıcı girdisinin sistem güvenlik ve uyum politikalarına uygunluk denetimi.",
            "options": [
                "guvenli_istek",            # Temiz, legal ve meşru kullanıcı isteği
                "prompt_injection",         # Sistem promptunu ezmeye çalışan enjeksiyon saldırıları
                "jailbreak_attempt",        # Ahlaki, etik ve kural dışı persona dayatmaları
                "pii_ve_kisisel_veri_ifsa", # TC Kimlik, Kredi kartı, şifre, gizli anahtar paylaşımı
                "zararli_ve_yasadisi_icerik",# Şiddet, nefret söylemi, yasa dışı faaliyet kılavuzu
                "sistem_komut_ve_kod_saldirisi" # Sunucuya ters kabuk (reverse shell), SQL injection
            ]
        }
    })


@dataclass
class TrainingConfig:
    # Model Omurgası (Türkçe ve 100+ dil için mmBERT omurgalı Laya)
    base_model_name: str = "convaiinnovations/laya-multilingual"
    fallback_model_name: str = "answerdotai/ModernBERT-large"
    
    max_length: int = 512
    batch_size: int = 32          # RTX PRO 6000 / 5090 / 4090 için optimize batch
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
