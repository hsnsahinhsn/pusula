# Türkçe Karar Modeli (Turkish Decision Model)

Bu depo; agentic iş akışlarında, yönlendirme katmanlarında (routing), araç seçiminde (tool calling) ve triyaj sistemlerinde kullanılmak üzere tasarlanmış **düşük gecikmeli (~10-15 ms) Evrensel Türkçe Tipli Karar Modeli (Universal Turkish Decision Model)** eğitim ve sentetik veri üretim altyapısını içerir.

---

## 🎯 Proje Hedefi & Mimari

Geleneksel üretken LLM'leri (GPT-4, Claude, Qwen vb.) her karar, güvenlik denetimi veya yönlendirme adımında çalıştırmak yüksek gecikme (500 ms - 2000 ms) ve yüksek maliyet yaratır. 

**Türkçe Karar Modeli Mimarisi:**
- **Omurga (Backbone):** Çok dilli encoder mimarisi (322M - 421M parametre).
- **Tipli Karar Başlığı (Typed Decision Head):** Tek bir matris geçişinde (single pass) şu 3 çıktıyı üretir:
  1. `choice`: Aday seçenekler (`options`) arasından en uygun kategorik karar.
  2. `score`: 0.0 - 1.0 arası kalibre edilmiş güven / öncelik / risk skoru.
  3. `action_required`: Boolean (true/false) eylem veya acil onay filtresi.
- **Hız:** GPU üzerinde **<15 ms** çıkarım gecikmesi.
- **Halüsinasyon Riski:** Sıfır (metin üretmez, doğrudan tensör sınıflandırması yapar).

---

## 🌐 32 Makro Alanı Kapsayan Evrensel Taksonomi

1. **Evrensel Ana Yönlendirici (Master Router)**
2. **Tüm Sınavlar:** SAT, ACT, ESAT, STEP, IMAT, BMAT, IELTS, TOEFL, PTE, GRE, GMAT, YKS (TYT/AYT), LGS, KPSS, ALES, DGS, TUS, DUS.
3. **Sağlık, Tıp & Klinik Triyaj:** 112 acil durum, dahiliye, çocuk, ortopedi, psikiyatri, tahlil/lab, ilaç etkileşimi.
4. **Hukuk & Adalet:** İçtihat/emsal karar, sözleşmeler, iş/tazminat, ceza, KVKK, resmi dilekçe taslağı.
5. **Bankacılık, Finans & Kripto:** Fraud/dolandırıcılık alarmı, kredi faizleri, BIST hisse/fonlar, kripto varlıklar, makroekonomi.
6. **Sigortacılık, Aktüerya & Sosyal Güvenlik:** SGK, emeklilik hesabı, prim günleri, DASK, kasko/trafik hasar tazminatı, BES.
7. **Yazılım, BT & Yapay Zeka:** Debug/hata ayıklama, kod refactor, SQL/veritabanı, Docker/K8s/DevOps, PyTorch/MLOps, API entegrasyonu.
8. **Temel Bilimler (İleri STEM):** İleri matematik, geometri ispatları, kuantum/klasik fizik, organik kimya, moleküler biyoloji/genetik.
9. **Mühendislik & İmalat:** Elektrik-elektronik, makine/mekatronik/CAD, inşaat/statik/deprem, endüstri, TSE/ISO standartları.
10. **E-Ticaret & Pazar Yerleri:** İade/cayma hakkı, kargo gecikmesi, mükerrer ödeme, stok/ürün sorgusu, pazar yeri operasyonları.
11. **Lojistik, Gümrük & Tedarik Zinciri:** Uluslararası navlun, konşimento, GTİP kodu tespiti, gümrükleme, antrepo ve soğuk zincir.
12. **Emlak, Gayrimenkul & Tapu:** Kira artış oranı, tahliye taahhütnamesi, tapu/kadastro, konut değerleme rayici.
13. **Otomotiv & Araç Bakımı:** Motor arıza lambası / OBD-II kodları, ekspertiz/tramer, kasko/trafik sigortası, periyodik bakım.
14. **Kariyer, İK & İstihdam:** ATS uyumlu CV/LinkedIn, mülakat simülasyonu, maaş pazarlığı, istifa/ihbar süreçleri.
15. **Seyahat, Turizm & Vize:** Schengen ve ABD vize evrakları, konsolosluk randevusu, uçak/otel rezervasyonu, gezi rotası.
16. **Gastronomi, Mutfak Sanatları & Gıda Güvenliği:** Yemek tarifleri, pişirme teknikleri, gıda bozulması/botulizm riski, alerjenler.
17. **Beslenme, Fitness & Spor:** Günlük kalori/makro hesabı, diyet programları, hipertrofi antrenman planı, sporcu takviyeleri.
18. **Ebeveynlik, Çocuk Gelişimi & Aile:** Bebek bakımı/beslenme, dil/motor gelişimi, ergenlik krizleri, ekran bağımlılığı.
19. **Tarım, Ziraat & Veterinerlik:** Evcil hayvan acil zehirlenme, kedi/köpek bakımı, bitki hastalıkları ve tarımsal sulama.
20. **Çevre, İklim & Temiz Enerji:** Çatı GES/Güneş enerjisi, karbon ayak izi, sıfır atık, ÇED raporları, enerji kimlik belgesi.
21. **Siyaset & Kamu Diplomasisi:** e-Devlet işlemleri, seçim/anayasa süreçleri, jeopolitik, tarafsız olgusal bilgi.
22. **Tarih, Arkeoloji & Kültürel Miras:** Osmanlı/Selçuklu, Cumhuriyet ve inkılaplar, Antik Anadolu, dünya savaşları tarihi.
23. **Dilbilim, Çeviri & Edebiyat:** Yeminli tercüme, yabancı dil öğrenimi, proofreading, yaratıcı metin yazarlığı.
24. **Sanat, Müzik, Sinema & Tiyatro:** Müzik teorisi/armoni, film tahlili/dramaturji, resim teknikleri, sanat tarihi.
25. **Oyun, E-Spor & Oyun Geliştirme:** Unreal/Unity script, oyun mekanikleri/level design, rekabetçi e-spor, FPS ayarları.
26. **Medya, Pazarlama & Tasarım:** SEO analizi, sosyal medya stratejisi, reklam kampanyaları (Google/Meta Ads), grafik tasarım.
27. **Moda, Tekstil & Kişisel Bakım:** Cilt bakımı aktif içerikler (retinol/C vitamini), saç dökülmesi, kumaş türleri, stil rehberi.
28. **Felsefe, Din, Etik & Sosyoloji:** Yapay zeka ve biyoetik ikilemleri, ilahiyat soruları, epistemoloji/ontoloji, toplum analizi.
29. **Psikolojik Kriz & Acil Destek:** İntihar/kendine zarar riski (182 / 112 Acil), aile içi şiddet (183 KADES), panik atak ilk yardım.
30. **Gündelik Sohbet (Chit-Chat):** Selamlaşma, hal-hatır, hafif sohbeti tanıyan ve ağır servislere gitmeyen vana.
31. **Kişisel Verimlilik & Asistanlık:** Takvim kurma, e-posta taslağı, yapılacaklar listesi, uzun notları özetleme.
32. **Güvenlik, Guardrails & Fallback:** Prompt injection, jailbreak, kredi kartı/TC PII ifşası ve anlamsız girdileri ayıklama.

---

## 📁 Proje Yapısı

```
turkish-decision-model/
├── config.py                  # 32 dikey alan, model ve eğitim hiperparametreleri
├── synthetic_generator.py     # Yerel Qwen LLM ile sentetik Türkçe JSONL veri üretici
├── model.py                   # PyTorch Türkçe Tipli Karar Modeli Mimarisi
├── train.py                   # bfloat16 + Flash Attention eğitim ve kalibrasyon döngüsü
├── serve.py                   # Düşük gecikmeli (<15 ms) FastAPI mikroservisi
├── agentic_router_example.py  # Agentic iş akışında kullanım örneği
├── requirements.txt           # Gerekli kütüphaneler
└── data/                      # train.jsonl, val.jsonl, test.jsonl
```

---

## 🚀 Kurulum ve Adım Adım Çalıştırma Rehberi

### 1. Adım: Sanal Ortam ve Kütüphaneler

```bash
git clone https://github.com/hsnsahinhsn/turkish-decision-model.git
cd turkish-decision-model

python -m venv venv
# Linux / WSL:
source venv/bin/activate
# Windows:
.\venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Adım: Yerel LLM Sunucusunun Başlatılması (Sentetik Veri İçin)

GPU sunucunuzda (RTX PRO 6000, 5090 veya 4090) **Qwen 3.8 27B** veya **Qwen 3.5 9B** modelini vLLM ile başlatın:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-9B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.7 \
  --max-model-len 4096
```

### 3. Adım: Sentetik Türkçe Veri Üretimi

23 dikey alanda doğrulanmış, yüksek kaliteli Türkçe JSONL verisi üretmek için:

```bash
python synthetic_generator.py
```
Bu script üretilen her örneği Pydantic ile şema kontrolünden geçirir, kopyaları eler ve `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl` dosyalarını hazırlar.

### 4. Adım: Model Eğitimi & Sıcaklık Kalibrasyonu

Veri üretimi tamamlandıktan sonra vLLM sürecini durdurup tüm GPU belleğini karar modeline tahsis edebilirsiniz:

```bash
python train.py
```
- GPU üzerinde `bfloat16` ve `FlashAttention-2` yeteneklerini kullanır (birkaç dakikada tamamlanır).
- Eğitim sonunda **Platt Scaling (Sıcaklık Kalibrasyonu)** yaparak aşırı güveni (overconfidence) engeller ve Brier Skorunu optimize eder.
- En iyi ağırlıklar `checkpoints/turkish_decision_model/best/` dizinine kaydedilir.

### 5. Adım: Çıkarım Servisi (Inference API)

```bash
python serve.py
```
FastAPI servisi `http://localhost:8080` üzerinde ayağa kalkar.

Örnek cURL isteği:
```bash
curl -X POST "http://localhost:8080/decide" \
     -H "Content-Type: application/json" \
     -d '{
       "context": "Faturamda fazla kesinti var, derhal iade talep ediyorum.",
       "options": ["bankacilik_finans_kripto", "e_ticaret_lojistik", "hukuk_adalet_mevzuat"]
     }'
```

Yanıt (<15 ms):
```json
{
  "choice": "bankacilik_finans_kripto",
  "choice_confidence": 0.9652,
  "score": 0.9410,
  "action_required": true,
  "latency_ms": 11.2
}
```

### 6. Adım: Ajanik Orkestrasyon Testi

```bash
python agentic_router_example.py
```
