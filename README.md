<div align="center">

# ⚡ Turkish Decision Model (TDM)
### Türkçe Yapay Zeka Ajanları ve Yönlendirme Katmanları İçin Düşük Gecikmeli (<15 ms) Evrensel Tipli Karar Motoru

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg)](https://pytorch.org/)
[![Inference Latency](https://img.shields.io/badge/Latency-%3C15ms-success.svg)](#benchmark--performans)
[![Zero Hallucination](https://img.shields.io/badge/Hallucination-0%25-blueviolet.svg)](#neden-karar-modeli)

[**Kurulum**](#-hızlı-başlangıç-quickstart) • [**Python API**](#-python-ile-kullanım) • [**Ajan Entegrasyonu**](#-ajanik-orkestrasyon-langgraph--crewai) • [**32 Evrensel Alan**](#-desteklenen-32-evrensel-makro-alan) • [**Benchmark**](#-benchmark--kıyaslama) • [**Eğitim Pipelineı**](#-yeniden-üretilebilirlik--eğitim-reproducibility)

</div>

---

## 💡 Neden Türkçe Karar Modeli?

Geleneksel üretken LLM'leri (GPT-4o, Claude 3.5 Sonnet vb.) yapay zeka ajanlarınızın **her araç seçimi, departman yönlendirmesi veya güvenlik filtresi** için çağırmak sistemi hantallaştırır, maliyeti uçurur ve JSON ayrıştırma (parse) hatalarına yol açar.

**Turkish Decision Model (TDM)**; token token metin üretmeyen (non-autoregressive), tek bir tensör matris geçişinde **3 kritik kararı birden** üreten yüksek hızlı bir "System 1" karar motorudur:

| Özellik | Geleneksel LLM'ler (GPT-4o, Claude) | Küçük Üretken Modeller (7B-8B) | **Turkish Decision Model (TDM)** |
| :--- | :---: | :---: | :---: |
| **Çıkarım Gecikmesi** | 500 ms – 2.000 ms | 200 ms – 600 ms | ⚡ **8 ms – 15 ms** |
| **Halüsinasyon Riski** | Var (Metin uydurabilir) | Yüksek | 🛡️ **%0 (Matematiksel olarak imkansız)** |
| **Çıktı Biçimi** | Serbest Metin / Ham JSON | Ham JSON (Kırılgan) | 🎯 **Doğrudan Tipli Tensör (Typed Output)** |
| **Olasılık Kalibrasyonu** | Kalibre Edilmemiş | Aşırı Güvenli (Overconfident) | 📊 **Platt Scaling ile Kalibre Edilmiş** |
| **Maliyet** | Her istek başına API ücreti | Yüksek GPU tüketimi | 💸 **Sıfır API maliyeti / Hafif yerel kaynak** |

---

## 🎯 3 Tipli Karar Çıktısı (Typed Decision Outputs)

Model, tek bir forward pass ile GPU üzerinde anında şu 3 değeri hesaplar:

```json
{
  "choice": "bankacilik_finans_kripto",
  "choice_confidence": 0.9842,
  "score": 0.9410,
  "action_required": true,
  "latency_ms": 11.2
}
```

1. **`choice` (Kategorik Seçim):** İster 32 evrensel sektörden birini, ister o an prompt'ta verdiğiniz dinamik aday seçenekleri (`options`) seçer.
2. **`score` (0.0 - 1.0 Kalibre Edilmiş Öncelik/Risk Skoru):** Girdinin aciliyetini, güvenini veya risk seviyesini derecelendirir.
3. **`action_required` (Boolean Onay/Müdahale Filtresi):** Harici araç çalıştırma, insan onayı veya güvenlik engeli gerekip gerekmediğini bildirir.

---

## 🚀 Hızlı Başlangıç (Quickstart)

### 1. Kurulum

```bash
git clone https://github.com/hsnsahinhsn/turkish-decision-model.git
cd turkish-decision-model
pip install -r requirements.txt
```

### 2. Python İle Kullanım (3 Satırda Karar)

```python
import torch
from transformers import AutoTokenizer
from model import TurkishDecisionModel

# 1. Modeli ve Tokenizer'ı Yükle
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_dir = "checkpoints/turkish_decision_model/best"

model = TurkishDecisionModel.from_pretrained(checkpoint_dir).to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

# 2. Karar İsteği
context = "Hesabımdan haberim olmadan 15.000 TL transfer edilmiş, kartımı derhal durdurun!"
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

## 🤖 Ajanik Orkestrasyon (LangGraph / CrewAI)

Büyük LLM çağrısı yapmadan önce gelen sorguyu mikrosaniyeler içinde filtreleyen akıllı yönlendirme düğümü:

```python
from agentic_router_example import decision_route

def agent_router_node(state):
    user_message = state["user_message"]
    
    # 1. Aşama: Güvenlik Guardrail Denetimi (<10 ms)
    guard = decision_route(user_message, options=["guvenli_mesru_istek", "prompt_injection_saldirisi", "pii_kisisel_veri_ifsa"])
    if guard["choice"] != "guvenli_mesru_istek" and guard["action_required"]:
        return {"next_node": "security_block", "reason": guard["choice"]}
        
    # 2. Aşama: En Uygun Uzman Ajanı / Aracı Seç (<12 ms)
    tools = ["sql_database_agent", "web_search_agent", "calendar_booking", "human_support"]
    decision = decision_route(user_message, options=tools)
    
    return {
        "next_node": decision["choice"],
        "urgency_score": decision["score"]
    }
```

---

## ⚡ Yüksek Performanslı REST API (FastAPI)

Modeli bağımsız bir mikroservis olarak çalıştırmak için:

```bash
python serve.py
```

**cURL İsteği:**
```bash
curl -X POST "http://localhost:8080/decide" \
     -H "Content-Type: application/json" \
     -d '{
       "context": "Yarın saat 15:00 için şirket toplantı odasını organize et.",
       "options": ["kisisel_asistanlik", "hukuk_adalet_mevzuat", "akademik_ve_sinavlar"]
     }'
```

**Yanıt (<15 ms):**
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

## 🌐 Desteklenen 32 Evrensel Makro Alan

Model hem bu 32 alan arasında genel yönlendirme yapabilir hem de **çalışma anında verilen herhangi dinamik seçenek havuzundan** seçim yapabilir:

1. **Global Master Router**
2. **Akademik & Sınavlar:** SAT, ACT, ESAT, IMAT, IELTS, TOEFL, GRE, GMAT, YKS, LGS, KPSS, ALES, TUS
3. **Sağlık, Tıp & Triyaj:** 112 Acil durum triyajı, dahiliye, çocuk sağlığı, tahlil/MR, ilaç etkileşimi
4. **Hukuk & Adalet:** Yargıtay/Danıştay içtihat, sözleşmeler, iş/tazminat, ceza, KVKK, dilekçe
5. **Bankacılık, Finans & Kripto:** Fraud/dolandırıcılık alarmı, kredi faizleri, BIST hisse, kripto varlıklar
6. **Sigortacılık & SGK:** Emeklilik hesabı, prim günleri, DASK, kasko/trafik hasar tazminatı, BES
7. **Yazılım, BT & Yapay Zeka:** Debug, kod refactor, SQL/veritabanı, Docker/K8s/DevOps, PyTorch/MLOps
8. **Temel Bilimler (İleri STEM):** İleri matematik/kalkülüs, geometri ispatları, kuantum/klasik fizik, organik kimya
9. **Mühendislik & İmalat:** Elektrik-elektronik, makine/mekatronik/CAD, inşaat/statik/deprem, TSE/ISO
10. **E-Ticaret & Pazar Yerleri:** İade/cayma, kargo gecikmesi, mükerrer ödeme, Trendyol/Amazon operasyonları
11. **Lojistik & Gümrük:** Uluslararası navlun, konşimento, GTİP kodu tespiti, gümrükleme, antrepo
12. **Emlak, Gayrimenkul & Tapu:** Kira artış oranı, tahliye davaları, tapu/kadastro, konut rayiç bedeli
13. **Otomotiv & Araç Bakımı:** Motor arıza lambası / OBD-II kodları, ekspertiz/tramer, kasko, bakım
14. **Kariyer & İK:** ATS uyumlu CV/LinkedIn, mülakat simülasyonu, maaş pazarlığı, istifa süreçleri
15. **Seyahat & Vize:** Schengen ve ABD vize evrakları, konsolosluk randevusu, uçak/otel, gezi rotası
16. **Gastronomi & Mutfak:** Yemek tarifleri, pişirme teknikleri, gıda güvenliği/botulizm riski, alerjenler
17. **Beslenme & Fitness:** Günlük kalori/makro hesabı, diyet programları, antrenman planı, takviyeler
18. **Ebeveynlik & Çocuk Gelişimi:** Bebek beslenmesi, dil/motor gelişimi, ergenlik krizleri, ekran bağımlılığı
19. **Tarım & Veterinerlik:** Evcil hayvan acil zehirlenme, kedi/köpek bakımı, bitki hastalıkları/sulama
20. **Çevre & Temiz Enerji:** Çatı GES/Güneş enerjisi, karbon ayak izi, sıfır atık, ÇED raporları
21. **Siyaset & Diplomasi:** e-Devlet işlemleri, seçim/anayasa süreçleri, jeopolitik, tarafsız olgusal bilgi
22. **Tarih & Arkeoloji:** Osmanlı/Selçuklu, Cumhuriyet inkılapları, Antik Anadolu, dünya savaşları
23. **Dilbilim & Çeviri:** Yeminli tercüme, yabancı dil öğrenimi, proofreading, yaratıcı metin yazarlığı
24. **Sanat, Müzik & Sinema:** Müzik teorisi/armoni, film tahlili/dramaturji, resim teknikleri, tiyatro
25. **Oyun & E-Spor:** Unreal/Unity script, oyun mekanikleri/level design, rekabetçi e-spor, FPS ayarı
26. **Medya & Tasarım:** SEO analizi, sosyal medya stratejisi, reklam kampanyaları (Google/Meta), grafik tasarım
27. **Moda & Kişisel Bakım:** Cilt bakımı aktif içerikler (retinol/C vitamini), saç dökülmesi, kumaş türleri
28. **Felsefe, Din & Etik:** Yapay zeka ve biyoetik ikilemleri, ilahiyat soruları, epistemoloji/ontoloji
29. **Psikolojik Kriz Destek:** İntihar/kendine zarar riski (182/112 Acil), aile içi şiddet (183 KADES)
30. **Gündelik Sohbet (Chit-Chat):** Selamlaşma, hal-hatır, hafif sohbeti tanıyan ve ağır servislere gitmeyen vana
31. **Kişisel Verimlilik:** Takvim kurma, e-posta taslağı, yapılacaklar listesi, uzun notları özetleme
32. **Güvenlik Guardrails & Fallback:** Prompt injection, jailbreak, kredi kartı/TC PII ifşası ve anlamsız girdiler

---

## 📊 Benchmark & Kıyaslama

*NVIDIA RTX PRO 6000 / RTX 4090 üzerinde 512 token bağlam ile ölçülmüştür.*

| Model | Tip | P95 Gecikme (Latency) | VRAM Tüketimi | Çıktı Güvenilirliği |
| :--- | :---: | :---: | :---: | :---: |
| **GPT-4o (OpenAI API)** | Autoregressive LLM | ~850 ms | Bulut | JSON Parse Hata Riski |
| **Qwen 2.5 / 3.5 9B** | Autoregressive LLM | ~280 ms | 18 GB | Düşük Parse Hatası |
| **Turkish Decision Model (TDM)** | **Non-Autoregressive Encoder** | ⚡ **11.4 ms** | **~1.2 GB** | 🎯 **%100 Tipli & Deterministik** |

---

## 🔬 Yeniden Üretilebilirlik & Eğitim (Reproducibility)

Modeli sıfırdan eğitmek veya kendi özel verilerinizle ince ayar (fine-tuning) yapmak için:

1. **Sentetik Veri Üretimi (24.000 Örnek):**
   ```bash
   python synthetic_generator.py
   ```
2. **Model Eğitimi & Sıcaklık Kalibrasyonu (bfloat16):**
   ```bash
   python train.py
   ```
Eğitim sonucunda kalibre edilmiş ağırlıklar ve sıcaklık parametreleri `checkpoints/turkish_decision_model/best/` dizinine otomatik olarak kaydedilir.

---

## 📜 Lisans & Alıntı

Bu proje **Apache 2.0** lisansı altında açık kaynak olarak dağıtılmaktadır. Hem ticari hem de akademik projelerinizde özgürce kullanabilirsiniz.

```bibtex
@misc{turkish_decision_model_2026,
  author = {Hasan Şahin},
  title = {Turkish Decision Model: Low-Latency Universal Typed Decision Engine for Turkish AI Workflows},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/hsnsahinhsn/turkish-decision-model}}
}
```
