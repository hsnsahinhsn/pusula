"""
Evrensel Türkçe Karar Modeli (Universal Turkish Decision Model) - 30+ Makro Alan Taksonomisi
Akademik sınavlar, tıp, hukuk, finans, yazılım, STEM, mühendislik, emlak, otomotiv, seyahat,
tarım, kariyer, felsefe, sohbet, kriz, gastronomi, sanat, oyun, çevre, sigorta/SGK, tarih,
lojistik, ebeveynlik ve moda/kişisel bakım dahil 30'dan fazla bağımsız makro alanı kapsar.
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
    
    # 30 dikey alan x 350 örnek = ~10.500 adet zengin, dengeli ve doğrulanmış sentetik veri
    samples_per_category: int = 350
    batch_size: int = 16
    output_dir: str = "data"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # 30+ MAKRO ALANDAN OLUŞAN EKSİKSİZ EVRENSEL KARAR TAKSONOMİSİ
    taxonomies: Dict[str, Dict] = field(default_factory=lambda: {
        # 1. EVRENSEL MAKRO YÖNLENDİRİCİ (Global Master Router)
        "global_master_routing": {
            "description": "Gelen kullanıcı sorgusunu 30 temel uzmanlık alanından en doğrusuna yönlendirme.",
            "options": [
                "akademik_ve_sinavlar",     # Sınavlar (SAT, ACT, ESAT, IMAT, YKS, IELTS vb.)
                "saglik_tip_ve_klinik",     # Tıp, klinik branşlar, tahlil, ilaç
                "hukuk_adalet_mevzuat",     # Yargı, dava, sözleşme, KVKK
                "bankacilik_finans_kripto", # Finans, borsa, kripto, vergi
                "sigorta_ve_sosyal_guvenlik", # SGK, emeklilik, prim, DASK, kasko
                "yazilim_bt_yapayzeka",     # Kodlama, hata ayıklama, veri tabanı, bulut
                "temel_bilimler_stem",      # İleri matematik, fizik, kimya, biyoloji
                "muhendislik_teknoloji",    # Elektrik, makine, inşaat, endüstri
                "e_ticaret_pazaryerleri",   # Sipariş, kargo, iade, mağaza yönetimi
                "lojistik_ve_gumruk",       # Navlun, konşimento, GTİP, antrepo
                "emlak_ve_gayrimenkul",     # Kira, tapu, değerleme, tahliye
                "otomotiv_ve_ulasim",       # Araç arızaları, ekspertiz, sigorta
                "kariyer_ve_ik",            # CV, mülakat, maaş pazarlığı, istifa
                "seyahat_turizm_vize",      # Vize başvurusu, konsolosluk, otel, rota
                "gastronomi_ve_mutfak",     # Tarifler, pişirme teknikleri, gıda güvenliği
                "beslenme_ve_fitness",      # Diyetisyen, antrenman, kalori, spor
                "ebeveynlik_cocuk_gelisimi",# Bebek gelişimi, ergenlik, aile psikolojisi
                "tarim_ve_veterinerlik",    # Zirai üretim, evcil hayvan acil zehirlenme
                "cevre_ve_yenilenebilir_enerji", # Güneş enerjisi, sıfır atık, karbon
                "siyaset_ve_diplomasi",     # Kamu politikası, e-Devlet, jeopolitik
                "tarih_ve_arkeoloji",       # Osmanlı, Cumhuriyet, Antik medeniyetler
                "dilbilim_ve_ceviri",       # Yeminli tercüme, yabancı dil öğrenimi
                "sanat_muzik_sinema",       # Müzik teorisi, sinema analizi, resim
                "oyun_espor_oyun_motoru",   # Unreal/Unity, oyun mekanikleri, e-spor
                "medya_pazarlama_tasarim",  # Dijital pazarlama, SEO, grafik tasarım
                "moda_ve_kisisel_bakim",    # Cilt bakımı, tekstil, stil danışmanlığı
                "felsefe_din_ve_etik",      # Etik ikilemler, ilahiyat, sosyoloji
                "psikolojik_kriz_destek",   # İntihar, şiddet, akut kriz (182/183/112 Acil!)
                "gundelik_sohbet_selam",    # Hal-hatır sorma, selamlama, chit-chat
                "kisisel_asistanlik",       # Takvim, e-posta taslağı, yapılacaklar listesi
                "belirsiz_anlamsiz_girdi",  # Out-of-domain fallback, anlamsız metin
                "guvenlik_ve_guardrails"    # Prompt injection, jailbreak, saldırı engelleme
            ]
        },

        # 2. ULUSAL & ULUSLARARASI TÜM SINAVLAR (All National & Global Exams)
        "exams_and_academics": {
            "description": "Tüm ulusal ve küresel üniversite, kabul ve dil sınavı kategorileri.",
            "options": [
                "yks_tyt_ayt_ydt",          # Türkiye üniversiteye giriş
                "lgs_ortaogretim",          # Liselere geçiş sınavı
                "kpss_kamu_personeli",      # KPSS Genel Yetenek/Kültür, Alan Bilgisi
                "ales_ve_dgs",              # Lisansüstü eğitim ve DGS sayısal/sözel
                "tus_ve_dus",               # Tıp ve Diş Hekimliği Uzmanlık Sınavları
                "sat_ve_act",               # ABD/Dünya üniversite kabul (Math & Reading)
                "esat_ve_step",             # Cambridge/Imperial mühendislik/fen sınavı
                "imat_ve_bmat",             # İtalya/Avrupa uluslararası tıp fakültesi sınavı
                "ielts_toefl_pte",          # Küresel İngilizce dil yeterlilik sınavları
                "gre_ve_gmat",              # Küresel MBA, yüksek lisans ve doktora kabulü
                "ydil_ve_yokdil",           # YDS, e-YDS ve YÖKDİL akademik dil sınavları
                "soru_cozumu_adim_adim",    # Formül ispatı ve adım adım soru çözümü
                "sinav_kocu_ve_strateji"    # Net artırma taktikleri ve çalışma planı
            ]
        },

        # 3. SAĞLIK, TIP & KLİNİK TRİYAJ (Health & Clinical)
        "health_and_triage": {
            "description": "Tıbbi semptom, acil triyaj ve klinik branş yönlendirmesi.",
            "options": [
                "acil_vaka_112_triyaj",     # Hayati tehlike, kalp krizi, bilinç kaybı (ÇOK ACİL!)
                "dahiliye_ve_genel_tip",    # Enfeksiyon, tansiyon, halsizlik, diyabet
                "pediyatri_cocuk_sagligi",  # Çocuk hastalıkları, aşı takvimi, ateş
                "ortopedi_ve_fizik_tedavi", # Eklem, kırık/çıkık, bel/boyun fıtığı
                "psikiyatri_ve_ruh_sagligi",# Anksiyete, panik atak, depresyon, terapi
                "laboratuvar_ve_tahlil",    # Kan/idrar tahlili, MR, Röntgen uyarısı
                "ilac_etkilesimi_ve_eczane" # Yan etki riski, doz aşımı kontrolü
            ]
        },

        # 4. HUKUK, ADALET & YASAL İŞLEMLER (Law & Legal)
        "law_and_justice": {
            "description": "Hukuki ihtilaf, dava ve mevzuat türünü belirleme.",
            "options": [
                "yargitay_danistay_ictihat",# Emsal karar ve içtihat tarama
                "sozlesmeler_hukuku",       # Ticari kira, satış, gizlilik sözleşmesi
                "is_ve_tazminat_hukuku",    # İşe iade, kıdem/ihbar, fazla mesai
                "ceza_ve_sorusturma",       # Savcılık şikayeti, suç duyurusu, ifade
                "aile_ve_miras_hukuku",     # Boşanma, velayet, veraset ilamı
                "kvkk_ve_bilisim_hukuku",   # Kişisel veri silme, siber suçlar
                "resmi_dilekce_taslagi",    # Mahkeme veya icra müdürlüğü dilekçesi
                "uzman_avukat_yonlendirme"  # Bağlayıcı yasal tavsiye için vekile aktarım
            ]
        },

        # 5. BANKACILIK, FİNANS, KRİPTO & EKONOMİ (Banking & Wealth)
        "banking_finance_crypto": {
            "description": "Finansal işlemler, yatırımlar ve piyasa analizlerini yönlendirme.",
            "options": [
                "supheli_islem_ve_fraud",   # Çalıntı kart, izinsiz EFT (Acil Eylem!)
                "kredi_faiz_ve_mevduat",    # Konut/ihtiyaç kredisi, vadeli mevduat faizi
                "bist_hisse_ve_fonlar",     # Borsa İstanbul, fon portföyü, temettü
                "kripto_ve_dijital_varlik", # Bitcoin, Ethereum, blockchain transferi
                "makroekonomi_ve_enflasyon",# TÜFE, TCMB faiz kararları, döviz kurları
                "fatura_vergi_ve_muhasebe", # Gelir vergisi, KDV iadesi, e-arşiv fatura
                "pos_ve_sirket_tahsilati"   # Ticari bankacılık, bloke, çek/senet
            ]
        },

        # 6. SİGORTACILIK, AKTÜERYA & SOSYAL GÜVENLİK (Insurance & Social Security)
        "insurance_and_social_security": {
            "description": "SGK, emeklilik hesabı, prim günleri, kasko ve sigorta tazminatları.",
            "options": [
                "sgk_ve_emeklilik_hesabi",  # EYT, prim gün sayısı, ne zaman emekli olurum
                "tamamlayici_saglik_sigortasi", # Anlaşmalı hastaneler, provizyon onayı
                "dask_deprem_sigortasi",    # Zorunlu deprem sigortası, poliçe yenileme
                "kasko_ve_trafik_hasar_dosyasi", # Kaza sonrası tazminat, değer kaybı davası
                "bireysel_emeklilik_bes",   # Devlet katkısı, fon getirisi, ayrılma şartları
                "hayat_ve_maluliyet_sigortasi" # Sürekli iş göremezlik tazminatı, vefat teminatı
            ]
        },

        # 7. YAZILIM, YAPAY ZEKA & SİBER GÜVENLİK (Software & AI)
        "software_ai_security": {
            "description": "Yazılım mühendisliği, sistem yönetimi ve yapay zeka görevleri.",
            "options": [
                "hata_ayiklama_debug",      # Runtime error, NullPointer, segfault çözümü
                "kod_yazma_ve_refactor",    # Temiz kod, modüler fonksiyon, mimari tasarım
                "veritabani_ve_sql",        # PostgreSQL, MySQL, NoSQL indeksleme ve sorgu
                "devops_docker_kubernetes", # CI/CD pipeline, container, Linux sunucu yönetimi
                "yapay_zeka_ve_mlops",      # PyTorch, LLM, model eğitimi, embedding, RAG
                "api_ve_mikroservis",       # FastAPI, Spring, gRPC, webhook entegrasyonu
                "siber_guvenlik_ve_pentest" # OWASP açık tarama, XSS, SQLi, yetkilendirme
            ]
        },

        # 8. TEMEL BİLİMLER (İLERİ STEM) (Fundamental Sciences)
        "stem_sciences_advanced": {
            "description": "Doğa bilimleri ve ileri matematiksel sorgular.",
            "options": [
                "ileri_matematik_ve_cebir", # Lineer cebir, diferansiyel denklemler, kalkülüs
                "geometri_ve_topoloji",     # Analitik geometri, vektörler, uzamsal ispatlar
                "klasik_ve_kuantum_fizik",  # Mekanik, termodinamik, rölativite, kuantum
                "organik_ve_fizikokimya",   # Kimyasal kinetik, reaksiyon mekanizmaları
                "molekuler_biyoloji_genetik",# DNA/RNA, hücresel solunum, enzim kinetiği
                "sayisal_hesaplama_araci",  # Sembolik matematik motoru (SymPy) çalıştırma
                "bilimsel_makale_ve_teori"  # Akademik literatür tarama ve teorem analizi
            ]
        },

        # 9. MÜHENDİSLİK & İMALAT (Applied Engineering)
        "applied_engineering": {
            "description": "Mühendislik disiplinleri ve endüstriyel teknik hesaplamalar.",
            "options": [
                "elektrik_elektronik_muh",  # Devre tasarımı, gömülü sistemler, mikroişlemciler
                "makine_ve_mekatronik_muh", # Mukavemet, akışkanlar mekaniği, CAD/CAM
                "insaat_ve_yapi_muh",       # Statik hesaplama, deprem yönetmeliği, betonarme
                "endustri_ve_verimlilik",   # Yöneylem araştırması, tedarik zinciri optimizasyonu
                "teknik_standartlar_tse_iso" # Kalite kontrol, CE/ISO sertifikasyon şartları
            ]
        },

        # 10. E-TİCARET & PAZAR YERLERİ (E-Commerce)
        "ecommerce_retail_marketplaces": {
            "description": "E-ticaret sipariş, teslimat ve pazar yeri (Trendyol/Amazon vb.) operasyonları.",
            "options": [
                "iade_degisim_ve_cayma",    # İade kargo kodu, ayıplı mal bildirimi
                "kargo_ve_lojistik_gecikme",# Transfer merkezinde takılan paket, kurye şikayeti
                "odeme_ve_tahsilat_uyusmazligi", # Karttan çekilip sipariş oluşmaması
                "urun_stok_ve_uyumluluk",   # Beden tablosu, orijinal ürün teyidi, teknik özellik
                "satici_ve_magaza_yonetimi",# Komisyon oranları, fatura yükleme, stok entegrasyonu
                "canli_insan_temsilcisine_devir" # Botun yetersiz kaldığı durumlarda müşteri temsilcisi
            ]
        },

        # 11. LOJİSTİK, GÜMRÜK & TEDARİK ZİNCİRİ (Logistics & Supply Chain)
        "logistics_and_customs": {
            "description": "Uluslararası taşımacılık, navlun, gümrükleme ve antrepo süreçleri.",
            "options": [
                "gumruk_mevzuati_ve_gtip",  # GTİP kodu tespiti, gümrük vergisi, ithalat/ihracat
                "denizyolu_ve_konnsimento", # Konşimento (Bill of Lading), konteyner takibi
                "havayolu_ve_iata_kargo",   # Hava kargo taşıma senedi (AWB), hızlı kurye
                "antrepo_ve_depolama",      # Serbest bölge, soğuk zincir, stok kontrolü
                "uluslararasi_navlun_teklifi" # Karayolu/denizyolu navlun maliyet hesabı
            ]
        },

        # 12. EMLAK, GAYRİMENKUL & TAPU (Real Estate)
        "real_estate_and_housing": {
            "description": "Kira, tapu, emlak değerleme ve konut mevzuatı.",
            "options": [
                "kira_artisi_ve_uyusmazlik",# TÜFE kira tavanı, tahliye taahhütnamesi
                "tapu_ve_kadastro_islemleri",# İpotek fekki, hisseli tapu, satış harcı
                "konut_degerleme_ve_ekspertiz", # Bölgesel metrekare rayiç bedeli hesabı
                "aidat_ve_apartman_yonetimi",# Kat Mülkiyeti Kanunu, ortak giderler
                "gayrimenkul_danismani_yonlendirme" # Emlak uzmanına doğrudan aktarım
            ]
        },

        # 13. OTOMOTİV, ARAÇ VE ULAŞIM (Automotive)
        "automotive_and_vehicles": {
            "description": "Araç alım-satım, ekspertiz, arıza kodları ve sigorta.",
            "options": [
                "ariza_lambasi_ve_obd_kodlari", # Motor arıza kodu (P0300 vb.), mekanik ses
                "ekspertiz_ve_hasar_kaydi", # Tramer sorgusu, boyalı parça, airbag kontrolü
                "ikinci_el_degerleme_piyasa", # Piyasa fiyat analizi, noter satış süreci
                "kasko_ve_trafik_sigortasi", # Kaza tutanağı, rücu davası, hasar dosya no
                "periyodik_bakim_ve_lastik" # Triger, yağ değişimi, kış lastiği zorunluluğu
            ]
        },

        # 14. KARİYER, İK & İSTİHDAM (Career & HR)
        "career_hr_employment": {
            "description": "Özgeçmiş, iş arama, mülakat hazırlığı ve işyeri hakları.",
            "options": [
                "cv_ve_linkedin_optimizasyonu", # ATS uyumlu özgeçmiş, önyazı (cover letter)
                "mulakat_simulasyonu_hazirlik", # Davranışsal ve teknik mülakat soruları
                "maas_ve_teklif_pazarligi", # Brüt-net maaş hesabı, yan haklar analizi
                "istifa_ve_ihbar_surecleri",# İstifa dilekçesi, ihbar süresi hesaplama
                "kariyer_donusumu_ve_mentorluk" # Alan değiştirme, sertifika ve eğitim tavsiyesi
            ]
        },

        # 15. SEYAHAT, TURİZM VE VİZE (Travel & Visa)
        "travel_tourism_visa": {
            "description": "Vize başvurusu, seyahat planlaması ve konaklama.",
            "options": [
                "schengen_ve_abd_vize_evraklari", # Banka dökümü, niyet mektubu, randevu
                "pasaport_ve_yurtdisi_cikis", # Harç pulu, pasaport yenileme, gümrük kuralları
                "ucus_otel_ve_rezervasyon", # Uçak bileti, bagaj hakkı, otel check-in
                "gezi_rotasi_ve_sehir_rehberi", # 3 günlük gezi planı, gezilecek yerler
                "iptal_ve_bilet_iade_hakki" # Rötar tazminatı, bilet açığa alma
            ]
        },

        # 16. GASTRONOMİ, MUTFAK SANATLARI & GIDA GÜVENLİĞİ (Gastronomy & Food)
        "gastronomy_and_culinary": {
            "description": "Tarifler, pişirme teknikleri, gıda saklama ve gıda zehirlenmesi uyarısı.",
            "options": [
                "yemek_tarifi_ve_malzemeler", # Türk ve dünya mutfağı reçeteleri
                "pisirme_teknikleri_ve_puf_noktasi", # Mühürleme, sous-vide, hamur mayalama
                "gida_guvenligi_ve_bozulma_riski", # Botulizm riski, SKT, çiğ tavuk/et saklama (Önemli!)
                "besin_alerjisi_ve_intolerans", # Gluten, laktoz, fıstık alerjisi uyarısı
                "restoran_ve_menu_yonetimi" # Maliyet hesabı, porsiyonlama, mutfak düzeni
            ]
        },

        # 17. BESLENME, FİTNESS VE SPOR (Nutrition & Fitness)
        "nutrition_fitness_sports": {
            "description": "Beslenme programı, kalori takibi ve spor antrenmanları.",
            "options": [
                "kalori_ve_makro_hesabi",   # Günlük kalori ihtiyacı, protein/karbonhidrat oranı
                "kilo_verme_ve_diyet_plani",# Ketojenik, aralıklı oruç, Akdeniz diyeti rehberi
                "fitness_antrenman_programi",# Hypertrophy, push-pull-legs, evde antrenman
                "sporcu_takviyesi_supplement", # Kreatin, whey protein, BCAA kullanımı
                "sakatlik_ve_esneme_uyarisi" # Kas ağrısı (DOMS), doğru form ve postür uyarısı
            ]
        },

        # 18. EBEVEYNLİK, ÇOCUK GELİŞİMİ & AİLE (Parenting & Child Development)
        "parenting_and_child_care": {
            "description": "Bebek bakımı, gelişim basamakları, ergen psikolojisi ve aile içi iletişim.",
            "options": [
                "bebek_bakimi_ve_beslenme", # Ek gıdaya geçiş, uyku eğitimi, kolik sancısı
                "cocuk_motor_ve_dil_gelisimi", # Konuşma gecikmesi, ince/kaba motor beceriler
                "ergenlik_ve_davranis_sorunlari", # Sınav kaygısı, öfke kontrolü, sınır koyma
                "ekran_bagimliligi_ve_teknoloji", # Dijital detoks, güvenli internet kullanımı
                "pedagojik_oyun_ve_etkinlik" # Zeka geliştirici oyunlar, Montessori aktiviteleri
            ]
        },

        # 19. TARIM, HAYVANCILIK VE VETERİNERLİK (Agriculture & Vet)
        "agriculture_and_veterinary": {
            "description": "Zirai üretim, bitki sağlığı ve evcil hayvan acil durumları.",
            "options": [
                "evcil_hayvan_acil_zehirlenme", # Çikolata, yabancı cisim yutma (Acil Veteriner!)
                "kedi_kopek_genel_bakim",   # Aşı takvimi, parazit, mama seçimi, tüy dökülmesi
                "bitki_hastaligi_ve_zararlilar", # Yaprak sararması, mantar, böcek ilacı
                "tarimsal_sulama_ve_gubre", # Damla sulama, azot-fosfor-potasyum dengesi
                "ciftlik_hayvani_sagligi"   # Büyükbaş/küçükbaş süt verimi ve hastalıkları
            ]
        },

        # 20. ÇEVRE, İKLİM & YENİLENEBİLİR ENERJİ (Environment & Energy)
        "environment_and_clean_energy": {
            "description": "Güneş enerjisi, sıfır atık, karbon ayak izi ve iklim mevzuatı.",
            "options": [
                "gunes_ve_ruzgar_enerjisi", # Çatı GES kurulumu, inverter, amortisman hesabı
                "karbon_ayak_izi_ve_surdurulebilirlik", # Yeşil mutabakat, sera gazı emisyonu
                "atik_yonetimi_ve_geri_donusum", # Sıfır atık belgesi, kompost, tehlikeli atık
                "cevre_mevzuati_ve_ced_raporu", # ÇED süreci, çevre izin ve lisansları
                "enerji_verimliligi_ve_tasarruf" # Binalarda ısı yalıtımı, enerji kimlik belgesi
            ]
        },

        # 21. SİYASET, KAMU YÖNETİMİ & DİPLOMASİ (Politics & Policy)
        "politics_public_diplomacy": {
            "description": "Kamu politikası, anayasal süreçler ve küresel diplomasi analizi.",
            "options": [
                "kamu_yonetimi_ve_edevlet",  # e-Devlet işlemleri, belediye hizmetleri
                "secim_ve_anayasa_surecleri",# Seçim kanunları, meclis denetimi
                "uluslararasi_iliskiler_jeopolitik", # BM, AB, NATO ilişkileri, sınır antlaşmaları
                "tarafsiz_siyasi_bilgi_sunumu", # Yansız, olgusal ve tarafsız perspektif
                "sosyoekonomik_kamu_reformu" # Sosyal güvenlik, eğitim ve sağlık reform analizleri
            ]
        },

        # 22. TARİH, ARKEOLOJİ & KÜLTÜREL MİRAS (History & Archaeology)
        "history_and_archaeology": {
            "description": "Osmanlı, Cumhuriyet, Antik medeniyetler ve dünya tarihi araştırmaları.",
            "options": [
                "osmanli_ve_selcuklu_tarihi",# Padişahlar, fermanlar, idari ve askeri yapı
                "cumhuriyet_tarihi_ve_inkilaplar", # Kurtuluş Savaşı, Lozan, Atatürk ilkeleri
                "antik_anadolu_medeniyetleri", # Hititler, Urartular, Frigler, kazı buluntuları
                "dunya_ve_avrupa_tarihi",   # Sanayi Devrimi, 1. ve 2. Dünya Savaşı, Soğuk Savaş
                "tarihi_belge_ve_kaynak_analizi" # Birincil kaynak inceleme, arşiv araştırması
            ]
        },

        # 23. DİL BİLİM, ÇEVİRİ & KÜLTÜR SANAT (Linguistics & Content)
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

        # 24. SANAT, MÜZİK, SİNEMA & SAHNE SANATLARI (Fine Arts & Entertainment)
        "arts_music_and_cinema": {
            "description": "Müzik teorisi, sinema analizi, resim teknikleri ve tiyatro.",
            "options": [
                "muzik_teorisi_ve_enstruman", # Armoni, akor dizilimleri, piyano/gitar tabları
                "film_ve_dizi_dramaturjisi", # Senaryo yapısı, sinematografi, film tahlili
                "gorsel_sanatlar_ve_resim", # Yağlı boya, perspektif, anatomi çizimi, sergi
                "sanat_tarihi_ve_akimlari", # Rönesans, Barok, Modernizm, Sürrealizm
                "tiyatro_ve_sahne_performansi" # Tirat, diksiyon, sahne tasarımı
            ]
        },

        # 25. OYUN, E-SPOR & OYUN MOTORLARI (Gaming & Esports)
        "gaming_esports_gamedev": {
            "description": "Oyun geliştirme (Unreal/Unity), mekanikler, donanım ve e-spor.",
            "options": [
                "oyun_motoru_unreal_unity", # C# / C++ script, Blueprint, fizik motoru
                "oyun_tasarimi_ve_mekanik", # Level design, dengeleme, boss dövüşü dinamikleri
                "espor_ve_rekabetci_oyunlar", # Taktik rehberi, meta analizi, takım stratejisi
                "oyun_donanimi_ve_fps_ayari", # Ekran kartı optimizasyonu, DLSS, input lag
                "masaustu_rpg_ve_kutu_oyunlari" # D&D karakter formu, kurallar, hikaye anlatımı
            ]
        },

        # 26. MEDYA, DİJİTAL PAZARLAMA & TASARIM (Media & Marketing)
        "media_marketing_design": {
            "description": "Dijital pazarlama, SEO, sosyal medya ve grafik tasarım.",
            "options": [
                "seo_ve_icerik_pazarlamasi",# Anahtar kelime analizi, arama motoru optimizasyonu
                "sosyal_medya_stratejisi",  # Instagram/LinkedIn içerik planı, reels fikirleri
                "reklam_kampanyasi_yonetimi",# Google Ads, Meta Ads hedef kitle optimizasyonu
                "grafik_ve_gorsel_tasarim", # Logo konsepti, renk paleti, tipografi istemi
                "video_kurgu_ve_podcast"    # YouTube script, podcast formatı, kurgu akışı
            ]
        },

        # 27. MODA, TEKSTİL & KİŞİSEL BAKIM (Fashion & Beauty)
        "fashion_and_personal_care": {
            "description": "Cilt bakımı, saç sağlığı, tekstil kumaşları ve stil danışmanlığı.",
            "options": [
                "cilt_bakimi_ve_aktif_icerik", # Retinol, C vitamini, asitler, bariyer onarımı
                "sac_bakimi_ve_dokulme",    # Saç mezoterapisi, dökülme karşıtı bakım, keratin
                "kumas_turleri_ve_tekstil", # Pamuk, keten, polyester, yıkama talimatı simgeleri
                "kisisel_stil_ve_kombin",   # Vücut tipine göre giyim, renk uyumu, gardırop
                "parfum_ve_koku_notalari"   # Notalar, kalıcılık, mevsime göre koku seçimi
            ]
        },

        # 28. FELSEFE, DİN, ETİK & SOSYOLOJİ (Humanities)
        "humanities_ethics_religion": {
            "description": "Etik ikilemler, ilahiyat, varoluşsal felsefe ve sosyoloji.",
            "options": [
                "etik_ikilemler_ve_ahlak",  # Yapay zeka etiği, biyoetik, vicdani sorumluluk
                "dini_ve_ilahiyat_sorulari",# İbadetler, fıkıh/fetva bilgisi, dinler tarihi
                "felsefi_akil_yurutme",     # Epistemoloji, ontoloji, varoluşçuluk, mantık
                "sosyoloji_ve_toplum_bilimi" # Toplumsal değişim, kültür, demografi analizi
            ]
        },

        # 29. PSİKOLOJİK KRİZ & ACİL DESTEK HATTI (Crisis Helpline - Critical Safety)
        "crisis_support_helpline": {
            "description": "Kendine zarar verme, şiddet, akut kriz tespiti (Hayati Güvenlik!).",
            "options": [
                "intihar_ve_kendine_zarar_riski", # 182 / 112 Acil Yardım hattı tetikleyici (Skor: 1.0)
                "aile_ici_siddet_ve_tehlike", # 183 Kadın Destek / KADES yönlendirmesi
                "akut_panik_ve_nefes_egzersizi", # Sakinleştirici ilk yardım yönlendirmesi
                "guvenli_tavsiye_destegi"   # Hayati risk olmayan genel moral desteği
            ]
        },

        # 30. GÜNDELİK SOHBET & SELAMLAMA (Chit-Chat & Casual)
        "chit_chat_and_greetings": {
            "description": "Gündelik selamlaşma, hal-hatır sorma ve hafif muhabbet.",
            "options": [
                "selamlama_ve_tanisma",     # Merhaba, günaydın, iyi akşamlar, selam
                "hal_hatir_ve_sohbet",      # Nasılsın, ne haber, günün nasıl geçiyor
                "tesekkur_ve_kapanis",      # Teşekkür ederim, eline sağlık, görüşürüz
                "espri_ve_hafif_sohbet"     # Şaka anlat, fıkra bilgin var mı, eğlenceli sohbet
            ]
        },

        # 31. KİŞİSEL VERİMLİLİK & ASİSTANLIK (Personal Productivity)
        "personal_productivity_tools": {
            "description": "Kişisel organizasyon, zaman yönetimi ve gündelik asistanlık.",
            "options": [
                "takvim_ve_hatirlatici_kur",# Toplantı ayarla, alarm/hatırlatıcı oluştur
                "e_posta_taslagi_hazirla",  # Kurumsal veya samimi e-posta yazma
                "yapilacaklar_listesi_todo",# Günlük görev planı ve önceliklendirme
                "ozetleme_ve_not_cikarma"   # Uzun bir metni maddeler halinde özetleme
            ]
        },

        # 32. SİSTEM GÜVENLİĞİ, SALDIRI VE BELİRSİZ GİRDİ (Security & Guardrails)
        "security_guardrails_fallback": {
            "description": "Prompt injection, jailbreak, kötü niyetli saldırı ve anlamsız girdiler.",
            "options": [
                "guvenli_mesru_istek",      # Temiz, legal ve standart kullanıcı girdisi
                "prompt_injection_saldirisi",# Sistem promptunu ezmeye çalışan saldırılar
                "jailbreak_etik_disi_talep", # Kötü niyetli persona/kuralları delme girişimi
                "pii_kisisel_veri_ifsa",    # Kredi kartı, şifre, kimlik ifşa riski
                "zararli_ve_yasadisi_icerik",# Şiddet, nefret suçu, yasa dışı faaliyet kılavuzu
                "belirsiz_anlamsiz_girdi"   # Anlamsız karakterler, fallback / netleştirme talebi
            ]
        }
    })


@dataclass
class TrainingConfig:
    # Model Omurgası (Türkçe ve 100+ dil için mmBERT çok dilli omurga)
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
    output_dir: str = "checkpoints/turkish_decision_model"
    save_total_limit: int = 2


@dataclass
class InferenceConfig:
    checkpoint_dir: str = "checkpoints/turkish_decision_model/best"
    device: str = "cuda"
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    temperature_scaling: float = 1.0
