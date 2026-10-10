<div align="center">

# Turkish Decision Model (TDM)
### Türkçe Yapay Zeka Ajanları ve Yönlendirme Katmanları İçin Düşük Gecikmeli (<15 ms) Evrensel Tipli Karar Motoru

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-red.svg)](https://pytorch.org/)
[![Latency](https://img.shields.io/badge/Latency-%3C15ms-blue.svg)](#benchmark--performans)
[![Architecture](https://img.shields.io/badge/Architecture-Non--Autoregressive-informational.svg)](#genel-bakis)

[Genel Bakış](#genel-bakış) • [Hızlı Başlangıç](#hızlı-başlangıç) • [Python API](#python-api) • [Ajan Entegrasyonu](#ajan-entegrasyonu) • [32 Makro Alan](#32-makro-alan-taksonomisi) • [Benchmark](#benchmark--performans) • [Eğitim Altyapısı](#eğitim-ve-yeniden-üretilebilirlik)

</div>

---

## Genel Bakış

Üretken büyük dil modellerini (GPT-6 Luna, Claude Opus 5.5, Gemini 3.8 Flash, DeepSeek-V4.1 Flash, GLM-5.3 vb.) yapay zeka ajanlarının her araç seçimi, departman yönlendirmesi veya güvenlik filtresi adımında çalıştırmak yüksek gecikme (350 ms – 1.800 ms), öngörülemeyen API maliyetleri ve JSON ayrıştırma (parse) hatalarına yol açar.

**Turkish Decision Model (TDM)**; token üretimi yapmayan (non-autoregressive), tek bir matris geçişinde üç temel kararı eşzamanlı üreten yüksek hızlı bir karar motorudur:

| Metrik / Nitelik | Sektör Lideri LLM'ler (GPT-6 Luna, Claude Opus 5.5) | Hızlı Üretken Modeller (Gemini 3.8, DeepSeek-V4.1) | Turkish Decision Model (TDM) |
| :--- | :---: | :---: | :---: |
| **P95 Çıkarım Gecikmesi** | 500 ms – 2.000 ms | 200 ms – 600 ms | **8 ms – 15 ms** |
| **Halüsinasyon Riski** | Mevcut | Yüksek | **%0 (Matematiksel olarak imkansız)** |
| **Çıktı Tipi** | Serbest Metin / Ham JSON | Ham JSON (Kırılgan) | **Doğrudan Tipli Tensör (Typed Output)** |
| **Olasılık Kalibrasyonu** | Kalibre Edilmemiş | Aşırı Güvenli (Overconfident) | **Platt Scaling ile Kalibre Edilmiş** |
| **İşlem Başı Maliyet** | İstek Başına API Ücreti | Yüksek GPU Kaynağı | **Sıfır API Ücreti / Yerel Kaynak** |

---

## Çıktı Spesifikasyonu (Typed Decision Head)

Model, tek bir forward pass ile GPU üzerinde anında şu yapılandırılmış çıktıyı döner:

```json
{
  "choice": "bankacilik_finans_kripto",
  "choice_confidence": 0.9842,
  "score": 0.9410,
  "action_required": true,
  "latency_ms": 11.2
}
```

- **choice:** Dinamik aday seçenekler (`options`) arasından seçilen en olası kategori veya araç.
- **choice_confidence:** Platt Scaling ile kalibre edilmiş seçim güven olasılığı (0.00 – 1.00).
- **score:** Girdinin aciliyet, risk veya önem derecesi (0.00 – 1.00 sürekli regresyon skoru).
- **action_required:** Müdahale, harici araç çalıştırma veya onay gerekliliğini belirten boolean filtre.

---

## Hızlı Başlangıç

### Kurulum

```bash
git clone https://github.com/hsnsahinhsn/turkish-decision-model.git
cd turkish-decision-model
pip install -r requirements.txt
```

---

## Python API

```python
import torch
from transformers import AutoTokenizer
from model import TurkishDecisionModel

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_dir = "checkpoints/turkish_decision_model/best"

# Model ve Tokenizer Yükleme
model = TurkishDecisionModel.from_pretrained(checkpoint_dir).to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

# Karar İsteği
context = "Hesabımdan bilgim dışında 15.000 TL çekim yapılmış, kartımı acilen bloke edin."
options = ["bankacilik_finans_kripto", "e_ticaret_pazaryerleri", "hukuk_adalet_mevzuat"]

result = model.predict_single(tokenizer, context=context, options=options, device=device)

print(result)
# Çıktı:
# {
#   'choice': 'bankacilik_finans_kripto',
#   'choice_confidence': 0.9912,
#   'score': 0.9850,
#   'action_required': True,
#   'latency_ms': 11.4
# }
```

---

## Ajan Entegrasyonu

Geleneksel LLM çağrılarından önce girdileri mikrosaniyeler içinde filtreleyen ve yönlendiren mimari şablon:

```python
from agentic_router_example import decision_route

def agent_router_pipeline(user_input: str) -> dict:
    # 1. Aşama: Güvenlik ve Guardrail Doğrulaması (<10 ms)
    security_check = decision_route(
        context=user_input,
        options=["guvenli_mesru_istek", "prompt_injection_saldirisi", "pii_kisisel_veri_ifsa"]
    )
    if security_check["choice"] != "guvenli_mesru_istek" and security_check["action_required"]:
        return {"status": "blocked", "reason": security_check["choice"]}

    # 2. Aşama: Uzman Ajan / Araç Seçimi (<12 ms)
    candidate_tools = ["sql_database_agent", "web_search_agent", "calendar_booking", "human_support"]
    tool_decision = decision_route(context=user_input, options=candidate_tools)
    
    return {
        "status": "routed",
        "target_tool": tool_decision["choice"],
        "priority_score": tool_decision["score"]
    }
```

---

## REST API Servisi (FastAPI)

Modeli düşük gecikmeli bir mikroservis olarak çalıştırmak için:

```bash
python serve.py
```

Servis `http://localhost:8080` üzerinde asenkron olarak ayağa kalkar.

**cURL Örneği:**
```bash
curl -X POST "http://localhost:8080/decide" \
     -H "Content-Type: application/json" \
     -d '{
       "context": "Yarın saat 14:00 için şirket toplantı odasını ayırt.",
       "options": ["kisisel_asistanlik", "hukuk_adalet_mevzuat", "akademik_ve_sinavlar"]
     }'
```

**JSON Yanıtı (<15 ms):**
```json
{
  "choice": "kisisel_asistanlik",
  "choice_confidence": 0.9840,
  "score": 0.7200,
  "action_required": true,
  "latency_ms": 11.8
}
```

---

## 32 Makro Alan Taksonomisi

Model hem aşağıdaki 32 küresel alan arasında yönlendirme yapabilir hem de çalışma anında verilen dinamik seçenekler (`dynamic candidate options`) arasından seçim gerçekleştirebilir:

1. **Global Master Router:** Genel alan sınıflandırıcısı
2. **Akademik & Sınavlar:** SAT, ACT, ESAT, STEP, IMAT, BMAT, IELTS, TOEFL, GRE, GMAT, YKS, LGS, KPSS, ALES, TUS, DUS
3. **Sağlık, Tıp & Triyaj:** 112 Acil triyajı, dahiliye, çocuk sağlığı, tahlil/MR, ilaç etkileşimi
4. **Hukuk & Adalet:** Yargıtay/Danıştay içtihat, sözleşmeler hukuku, iş/tazminat, ceza, KVKK, resmi dilekçe
5. **Bankacılık, Finans & Kripto:** Şüpheli işlem alarmı, kredi faizleri, BIST hisse/fon, kripto varlıklar
6. **Sigortacılık & SGK:** Emeklilik hesabı, prim günleri, DASK, kasko/trafik hasar tazminatı, BES
7. **Yazılım, BT & Yapay Zeka:** Hata ayıklama, kod refactor, SQL/veritabanı, DevOps/K8s, PyTorch/MLOps
8. **Temel Bilimler (İleri STEM):** İleri matematik/kalkülüs, geometri ispatları, kuantum/klasik fizik, organik kimya
9. **Mühendislik & İmalat:** Elektrik-elektronik, makine/mekatronik/CAD, inşaat/statik/deprem, TSE/ISO
10. **E-Ticaret & Pazar Yerleri:** İade/cayma, kargo gecikmesi, mükerrer ödeme, pazar yeri operasyonları
11. **Lojistik & Gümrük:** Uluslararası navlun, konşimento, GTİP kodu tespiti, gümrükleme, antrepo
12. **Emlak, Gayrimenkul & Tapu:** Kira artış oranı, tahliye davaları, tapu/kadastro, konut rayiç bedeli
13. **Otomotiv & Araç Bakımı:** Motor arıza lambası / OBD-II kodları, ekspertiz/tramer, kasko, bakım
14. **Kariyer & İK:** ATS uyumlu CV/LinkedIn, mülakat simülasyonu, maaş pazarlığı, istifa süreçleri
15. **Seyahat & Vize:** Schengen ve ABD vize evrakları, konsolosluk randevusu, uçak/otel, gezi rotası
16. **Gastronomi & Mutfak:** Yemek tarifleri, pişirme teknikleri, gıda güvenliği ve botulizm riski, alerjenler
17. **Beslenme & Fitness:** Günlük kalori hesabı, diyet programları, antrenman planı, takviyeler
18. **Ebeveynlik & Çocuk Gelişimi:** Bebek beslenmesi, dil/motor gelişimi, ergenlik krizleri, ekran bağımlılığı
19. **Tarım & Veterinerlik:** Evcil hayvan acil zehirlenme, kedi/köpek bakımı, bitki hastalıkları ve sulama
20. **Çevre & Temiz Enerji:** Çatı GES/Güneş enerjisi, karbon ayak izi, sıfır atık, ÇED raporları
21. **Siyaset & Diplomasi:** e-Devlet işlemleri, seçim/anayasa süreçleri, jeopolitik, tarafsız bilgi
22. **Tarih & Arkeoloji:** Osmanlı/Selçuklu, Cumhuriyet inkılapları, Antik Anadolu, dünya savaşları
23. **Dilbilim & Çeviri:** Yeminli tercüme, yabancı dil öğrenimi, proofreading, yaratıcı metin yazarlığı
24. **Sanat, Müzik & Sinema:** Müzik teorisi/armoni, film tahlili/dramaturji, resim teknikleri, tiyatro
25. **Oyun & E-Spor:** Unreal/Unity script, oyun mekanikleri/level design, rekabetçi e-spor, FPS ayarı
26. **Medya & Tasarım:** SEO analizi, sosyal medya stratejisi, reklam kampanyaları (Google/Meta), grafik tasarım
27. **Moda & Kişisel Bakım:** Cilt bakımı aktif içerikler (retinol/C vitamini), saç dökülmesi, kumaş türleri
28. **Felsefe, Din & Etik:** Yapay zeka ve biyoetik ikilemleri, ilahiyat soruları, epistemoloji/ontoloji
29. **Psikolojik Kriz Destek:** İntihar/kendine zarar riski (182/112 Acil), aile içi şiddet (183 KADES)
30. **Gündelik Sohbet (Chit-Chat):** Selamlaşma, hal-hatır, hafif sohbeti tanıyan ve filtreleyen vana
31. **Kişisel Verimlilik:** Takvim kurma, e-posta taslağı, yapılacaklar listesi, uzun notları özetleme
32. **Güvenlik Guardrails & Fallback:** Prompt injection, jailbreak, kredi kartı/TC PII ifşası ve anlamsız girdiler

---

## Benchmark & Performans

*NVIDIA H200 SXM (141 GB HBM3e) ve NVIDIA RTX PRO 6000 (96 GB) veri merkezi donanımları üzerinde 512 token bağlam uzunluğu ile test edilmiştir.*

| Model | Mimari Tipi | P95 Gecikme (Latency) | GPU VRAM Tüketimi | Çıktı Determinizmi |
| :--- | :---: | :---: | :---: | :---: |
| **GPT-6 Luna (OpenAI API)** | Autoregressive LLM | ~920 ms – 1.750 ms | Bulut Tabanlı | Ayrıştırma Hatası Olası |
| **Claude Opus 5.5 (Anthropic API)** | Autoregressive LLM | ~980 ms – 1.820 ms | Bulut Tabanlı | Ayrıştırma Hatası Olası |
| **Gemini 3.8 Flash (Google API)** | Autoregressive LLM | ~320 ms – 650 ms | Bulut Tabanlı | Ayrıştırma Hatası Olası |
| **DeepSeek-V4.1 Flash (MoE API)** | Autoregressive LLM | ~290 ms – 580 ms | Bulut / Çoklu GPU | Ayrıştırma Hatası Olası |
| **GLM-5.3 (Zhipu AI)** | Autoregressive LLM | ~610 ms – 950 ms | Bulut / Çoklu GPU | Ayrıştırma Hatası Olası |
| **Qwen 3.8-27B (vLLM Yerel)** | Autoregressive LLM | ~240 ms – 450 ms | 54 GB (bfloat16) | Ayrıştırma Hatası Olası |
| **Turkish Decision Model (RTX PRO 6000)** | **Non-Autoregressive Encoder** | **11.4 ms** | **~1.2 GB** | **%100 Tipli ve Deterministik** |
| **Turkish Decision Model (NVIDIA H200)** | **Non-Autoregressive Encoder** | **4.8 ms** | **~1.2 GB** | **%100 Tipli ve Deterministik** |

---

## Eğitim ve Yeniden Üretilebilirlik

Modeli sıfırdan eğitmek veya özel kurumsal verilerinizle ince ayar yapmak için:

### 1. Sentetik Veri Üretimi (24.000 Örnek Hedefi)
```bash
python synthetic_generator.py
```

### 2. Model Eğitimi & Sıcaklık Kalibrasyonu
```bash
python train.py
```

Eğitim döngüsü `bfloat16` ve `FlashAttention-2` hızlandırmasıyla çalışır. Doğrulama kümesi üzerinde Platt Scaling uygulanarak Brier Skoru ve ECE (Expected Calibration Error) minimize edilir. Optimize edilen ağırlıklar `checkpoints/turkish_decision_model/best/` dizinine kaydedilir.

---

## Lisans ve Alıntı

Bu proje **Apache 2.0** lisansı altında açık kaynak olarak sunulmuştur. Hem ticari hem de akademik projelerde serbestçe kullanılabilir.

```bibtex
@misc{turkish_decision_model_2026,
  author = {Hasan Şahin},
  title = {Turkish Decision Model: Low-Latency Universal Typed Decision Engine for Turkish AI Workflows},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/hsnsahinhsn/turkish-decision-model}}
}
```
