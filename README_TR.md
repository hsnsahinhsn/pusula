<div align="center">

<h1><picture><source media="(prefers-color-scheme: dark)" srcset="site/brand/pusula-logo-reverse.svg"><img src="site/brand/pusula-logo.svg" alt="Pusula" width="300"></picture></h1>

### İlk Türkçe tam donanımlı karar modeli

[English](README.md) • [Türkçe](README_TR.md) • [Web sitesi](https://hsnsahinhsn.github.io/pusula/tr/)

[![CI](https://github.com/hsnsahinhsn/pusula/actions/workflows/ci.yml/badge.svg)](https://github.com/hsnsahinhsn/pusula/actions)
[![Lisans: Apache 2.0](https://img.shields.io/badge/Lisans-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)

</div>

Karar modelleri, bir metin hakkında tanımlı sorulara (örneğin bir kaydın hangi ekibe yönlendirileceği, metnin bir
prompt enjeksiyonu içerip içermediği ya da ne kadar acil olduğu) cevap verir ve metin üretmek yerine **her seçenek
için bir olasılık** döndürür. Mevcut karar modelleri (Decision 3.0, Jev, Laya ve diğerleri) İngilizce için
geliştirilmiştir ve Türkçedeki doğrulukları sınırlıdır. **Pusula, Türkçeye uyarlanmış Decision 3.0'dır.**

- **Önce Türkçe.** [vllm-sr/d3](https://huggingface.co/vllm-sr/d3) (Decision 3.0); bankacılık, muhasebe ve vergi,
  hukuk, sağlık, e-ticaret, lojistik ve kamu hizmetleri dahil 20 alandan Türkçe karar verisiyle ince ayarlanmıştır.
  Gündelik yazım, yazım hataları, eksik Türkçe karakterler ve Türkçe–İngilizce karışık metinler kapsanmaktadır.
- **İstek anında tanımlanan seçenekler.** Sorular ve seçenekleri her istekte ayrıca belirtilir.
- **Üç soru tipi.** `choice` (tek seçenek), `noul` (evet/hayır) ve `score` (sıralı seviyeler).
- **d3 ile doğrudan değiştirilebilir.** d3'ün `/v1/systemone` istek ve cevap biçimi aynen korunmuştur.
- **Hazır Türkçe taksonomi.** 20 alan, 86 alt alan ve 299 niyetin yanı sıra altı evrensel kontrol (aciliyet, eylem
  gerekliliği, insana devir, güvenlik riski, duygu ve kişisel veri) sunulmaktadır.
- **Türkçe karar benchmark'ı.** Pusula, bağımsız olarak etiketlenmiş Türkçe test setlerinde ham Decision 3.0 ile
  karşılaştırılmıştır; sonuçlar modelle birlikte yayımlanmaktadır.

> **Durum:** v1.0. Ağırlıklar: [Pusula](https://huggingface.co/hsnsahinhsn/pusula) ·
> [Pusula Nano](https://huggingface.co/hsnsahinhsn/pusula-nano).

## Modeller

| Model | Boyut | Çalıştığı yer | Benchmark / altın set | Ne için |
|---|---|---|---|---|
| **Pusula** (Türkçe Decision 3.0) | 26B: `vllm-sr/d3` üzerine 0,9 GB'lık LoRA adaptörü | GPU: BF16'da ≥ 64 GB, 4-bit'te ≥ 24 GB | **92,1 / 90,6** | En yüksek doğruluk; hazır ve özel sorular için |
| **Pusula Nano** | 149M ([TabiBERT](https://huggingface.co/boun-tabilab/TabiBERT) cross-encoder) | CPU veya herhangi bir GPU | 82,1 / 84,3 | Hazır sorularda yüksek hacim ve düşük gecikme |

Her iki model de aynı istekleri kabul eder.

---

## Sonuçlar

Doğruluk (%). Değerlendirme etiketleri, birbirinden bağımsız ve kör çalışan iki etiketçi (Claude Opus 5.5 ve
Claude Sonnet 5.5) tarafından atanmış; ayrışmalar hakem tarafından karara bağlanmıştır. Tüm etiketler yayımlanan
[etiketleme politikasına](pusula/data/label_policy.md) uygundur. Gerçek müşteri mesajları, aynı politika
çerçevesinde Claude tarafından kör olarak etiketlenmiştir. Değerlendirme etiketleri yalnızca ölçüm amacıyla
kullanılmakta olup hiçbir zaman eğitim etiketi olarak kullanılmamaktadır.

| Değerlendirme | Soru | Decision 3.0 | **Pusula** | Pusula Nano |
|---|---|---|---|---|
| Gerçek müşteri mesajları (500, eğitim dışı, lisanslı) | 4.500 | 77,4 | **84,1** | 82,9 |
| [Türkçe Karar Benchmark'ı](benchmark/README.md) (2.194 mesaj) | 19.861 | 88,5 | **92,1** | 82,1 |
| Altın test seti (eğitimde olmayan 759 mesaj) | 7.015 | 85,5 | **90,6** | 84,3 |
| Kendi sorularınız (497 mesaj × 3 özel soru) | 1.491 | **94,8** | 94,7 | 86,8 |
| MASSIVE tr-TR, 60 seçenekli niyet, sıfır örnek (insan yazımı) | 2.974 | **85,3** | 85,1 | 36,1 |
| OffensEval 2020 Türkçe, saldırgan dil (gerçek tweet'ler) | 3.528 | 88,4 | **88,6** | 37,2 |

Altın test seti, soru bazında:

| Soru | Decision 3.0 | **Pusula** |
|---|---|---|
| Aciliyet (4 seviye) | 53,2 | **81,0** |
| İnsana devretme | 81,5 | **88,4** |
| Alan | 82,1 | **87,0** |
| Duygu | 85,0 | **87,2** |
| Güvenlik riski | 94,2 | **97,9** |
| Eylem gerekli mi | 88,9 | **90,6** |
| Kişisel veri | 96,4 | 96,4 |
| Niyet | 96,8 | 96,8 |

- **Gerçek müşteri mesajları:** Değerlendirme, ticari bir iş ortağından lisanslanan ve eğitimde kullanılmayan 500
  anonim müşteri mesajı (yorum, şikâyet ve soru) üzerinde yapılmıştır. Bu set yayımlanmamaktadır. Aynı kaynaktan
  alınan diğer mesajlar eğitimde kullanılmış ve açık bir model (Qwen) tarafından etiketlenmiştir.
- **Aciliyet kısmen özneldir:** iki etiketçi mesajların %84,4'ünde aynı cevabı vermektedir; bu nedenle 81,0'lık
  sonuç ulaşılabilir tavana yakındır.
- **Özel sorular:** Pusula, Decision 3.0 ile aynı düzeydedir (94,7'ye 94,8; altın setteki özel sorularda +2,7,
  OffensEval'de +0,2, MASSIVE'de −0,2).
- **Pusula Nano** ağırlıklı olarak hazır sorularla eğitilmiştir. Gerçek mesajlardaki doğruluğu Pusula'ya yakındır
  (82,9'a 84,1). Hazır sorulardan farklı ya da çok seçenekli görevler için Pusula önerilmektedir (OffensEval'de
  88,6'ya 37,2, MASSIVE'de 85,1'e 36,1).
- MASSIVE ve OffensEval eğitim verisinde yer almamaktadır.

## İçindekiler

- [Kurulum](#kurulum)
- [Temel kavramlar](#temel-kavramlar)
- [Python API](#python-api)
- [HTTP API](#http-api)
- [Hazır taksonomi ve yönlendirme](#hazır-taksonomi-ve-yönlendirme)
- [İyi soru yazmak](#iyi-soru-yazmak)
- [Güven değerini kullanmak](#güven-değerini-kullanmak)
- [Örnekler](#örnekler)
- [Kendi modelinizi eğitmek](#kendi-modelinizi-eğitmek)
- [Katkıda bulunanlar ve atıf](#katkıda-bulunanlar-ve-atıf)
- [Lisans ve teşekkürler](#lisans-ve-teşekkürler)

---

## Kurulum

```bash
pip install "pusula[serve,d3] @ git+https://github.com/hsnsahinhsn/pusula"
hf download vllm-sr/d3 --local-dir ~/d3 --exclude "assets/*"     # temel Decision 3.0 ağırlıkları (~52 GB)
hf download hsnsahinhsn/pusula --local-dir ~/pusula                # Pusula adaptörü (~0,9 GB)
# ya da temel ağırlık gerektirmeyen küçük model:
hf download hsnsahinhsn/pusula-nano --local-dir ~/pusula-nano      # Pusula Nano (~0,6 GB)
```

`serve` eki PyTorch, Transformers ve FastAPI'yi; `d3` eki ise 4-bit yükleme için PEFT ve bitsandbytes'ı kurar.
PyTorch'un CUDA sürümünün önceden kurulması gerekir ([pytorch.org](https://pytorch.org/get-started/locally/)).
Pusula Nano için yalnızca `serve` eki yeterlidir.

## Temel kavramlar

Bir istek bir **durum** (`state`) ve adlandırılmış **sorulardan** (`questions`) oluşur:

```json
{
  "state": "Kartımdan bilgim dışında 4.500 TL çekilmiş, hemen bloke edin!",
  "questions": {
    "ekip": {
      "type": "choice",
      "instructions": "Bu talebi hangi ekip ele almalı?",
      "criteria": {
        "dolandiricilik": "Şüpheli işlem ve dolandırıcılık",
        "kart_islemleri": "Kart başvurusu, limit ve şifre",
        "genel_destek": "Diğer sorular"
      }
    },
    "acil": { "type": "noul", "instructions": "Hemen müdahale gerekiyor mu?" },
    "risk": {
      "type": "score",
      "instructions": "Finansal risk ne düzeyde?",
      "criteria": ["Düşük", "Orta", "Yüksek"]
    }
  }
}
```

| Tip | `criteria` | Cevap |
|---|---|---|
| `choice` | Seçenek anahtarlarını açıklamalara eşleyen nesne (`null` olabilir) | `choice`, `probabilities`, `confidence` |
| `noul` | İsteğe bağlı; `"false"` ve `"true"`yu açıklayabilir | `noul` = *evet* olasılığı |
| `score` | Düşükten yükseğe sıralı 2–10 seviyelik liste | `score` = beklenen seviye (0…n-1), `probabilities`, `confidence`, `legend` |

Yukarıdaki isteğin cevabı aşağıda gösterilmiştir (değerler örnek amaçlıdır):

```json
{
  "answers": {
    "ekip": {
      "type": "choice",
      "choice": "dolandiricilik",
      "probabilities": {"dolandiricilik": 0.94, "kart_islemleri": 0.05, "genel_destek": 0.01},
      "confidence": 0.94
    },
    "acil": {"type": "noul", "noul": 0.97},
    "risk": {
      "type": "score",
      "score": 1.86,
      "probabilities": {"0": 0.02, "1": 0.10, "2": 0.88},
      "confidence": 0.88,
      "legend": {"0": "Düşük", "1": "Orta", "2": "Yüksek"}
    }
  },
  "model": "pusula",
  "latency_ms": 9.8
}
```

## Python API

```python
from pusula.inference import D3Engine, DecisionEngine

engine = D3Engine.load("~/d3", adapter="~/pusula")      # Pusula (Türkçe Decision 3.0)
# engine = DecisionEngine.load("~/pusula-nano")          # Pusula Nano (GPU veya CPU)

answers = engine.decide(
    "Siparişim 5 gündür kargoda, hâlâ gelmedi. Artık iptal etmek istiyorum.",
    {
        "niyet": {
            "type": "choice",
            "instructions": "Müşteri ne istiyor?",
            "criteria": {
                "kargo_takip": "Gönderinin nerede olduğunu öğrenmek",
                "iptal": "Siparişi iptal etmek",
                "iade": "Teslim alınan ürünü iade etmek",
            },
        },
        "temsilci": {"type": "noul", "instructions": "Bir insan temsilciye aktarılmalı mı?"},
    },
)
print(answers["niyet"]["choice"], answers["niyet"]["confidence"])
print(answers["temsilci"]["noul"])
```

Birden çok istek tek çağrıda işlenebilir; istekler mümkün olan en az sayıda ileri geçişe yerleştirilir:

```python
results = engine.decide_many([(metin, sorular) for metin in mesajlar])
```

Sorular, yukarıdaki gibi sözlük olarak ya da `pusula.Question` nesnesi olarak verilebilir.

## HTTP API

Sunucu aşağıdaki gibi başlatılır:

```bash
PUSULA_D3_BASE=~/d3 PUSULA_MODEL_DIR=path/to/pusula pusula-serve --port 8080   # Pusula
PUSULA_MODEL_DIR=path/to/pusula-nano pusula-serve --port 8080                  # Pusula Nano
```

Sunucu varsayılan olarak yalnızca `127.0.0.1` üzerinde dinler. Başka makinelerden erişim için bir API anahtarı
tanımlanmalı ve `--host 0.0.0.0` verilmelidir; istemciler bu durumda her `/v1/*` isteğine `Authorization: Bearer
<anahtar>` (ya da `X-API-Key: <anahtar>`) başlığını ekler:

```bash
PUSULA_API_KEY=degistirin PUSULA_MODEL_DIR=path/to/pusula-nano pusula-serve --host 0.0.0.0 --port 8080
```

| Metot | Yol | Açıklama |
|---|---|---|
| `POST` | `/v1/systemone` | Tek `{state, questions}` isteği (d3 uyumlu) |
| `POST` | `/v1/systemone/batch` | `{"requests": [...]}`, en fazla 64 istek |
| `POST` | `/v1/route` | Mesajı hazır taksonomide yönlendirir |
| `GET` | `/v1/taxonomy` | Hazır taksonominin JSON hâli |
| `GET` | `/health` | Canlılık; her zaman 200 |
| `GET` | `/ready` | Model yüklenince 200, öncesinde 503 |

```bash
curl -s localhost:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Faturama kullanmadığım bir paket için 450 TL yansıtılmış.",
  "questions": {
    "departman": {"type": "choice", "instructions": "Hangi departman?",
                  "criteria": {"fatura": "Fatura ve ödeme", "teknik": "Arıza ve teknik destek"}},
    "iade": {"type": "noul", "instructions": "Müşteri para iadesi istiyor mu?"}
  }
}'
```

Geçersiz sorular HTTP 422 ve gerekçesiyle reddedilir. Sunucu, ortam değişkenleriyle yapılandırılır:

| Değişken | Varsayılan | Anlamı |
|---|---|---|
| `PUSULA_D3_BASE` | — | Temel `vllm-sr/d3` dizini; Pusula için gereklidir (bu durumda `PUSULA_MODEL_DIR` adaptörü gösterir) |
| `PUSULA_MODEL_DIR` | — | Pusula adaptör dizini ya da Pusula Nano model dizini |
| `PUSULA_DEVICE` | otomatik | `cuda`, `cuda:1`, `cpu`… |
| `PUSULA_CORS_ORIGINS` | yok | Tarayıcı istemcileri için virgülle ayrılmış izinli origin'ler |
| `PUSULA_MAX_QUESTIONS` | 32 | İstek başına soru sayısı |
| `PUSULA_MAX_BATCH` | 64 | Toplu çağrıdaki istek sayısı |
| `PUSULA_MAX_STATE_CHARS` | 60000 | En uzun durum metni |

## Hazır taksonomi ve yönlendirme

`POST /v1/route` (ya da `engine.route(metin)`) mesajı üç adımda (**alan**, **alt_alan** ve **niyet**) yönlendirir ve
evrensel soruları aynı çağrıda cevaplar:

```json
{
  "path": "finans_bankacilik.itiraz_ve_dolandiricilik.supheli_islem",
  "answers": {
    "alan": {"choice": "finans_bankacilik", "confidence": 0.97, "...": "..."},
    "aciliyet": {"type": "score", "score": 2.8, "...": "..."},
    "insana_devret": {"type": "noul", "noul": 0.71}
  }
}
```

| Alan anahtarı | Alan | Alt alan | Niyet |
|---|---|---:|---:|
| `finans_bankacilik` | Finans ve Bankacılık | 6 | 26 |
| `muhasebe_vergi` | Muhasebe ve Vergi | 5 | 20 |
| `hukuk_uyum` | Hukuk ve Mevzuat Uyumu | 5 | 20 |
| `saglik` | Sağlık | 5 | 15 |
| `sigorta_sosyal_guvenlik` | Sigorta ve Sosyal Güvenlik | 3 | 12 |
| `ticaret_eticaret` | Ticaret ve E-Ticaret | 5 | 19 |
| `lojistik_tedarik` | Lojistik ve Tedarik Zinciri | 5 | 21 |
| `abonelik_hizmetleri` | Telekom ve Enerji Abonelikleri | 3 | 13 |
| `teknoloji_bt` | Teknoloji ve Bilgi İşlem | 6 | 22 |
| `insan_kaynaklari` | İnsan Kaynakları | 5 | 17 |
| `kamu_hizmetleri` | Kamu Hizmetleri | 4 | 13 |
| `egitim` | Eğitim | 4 | 13 |
| `seyahat_konaklama` | Seyahat ve Konaklama | 4 | 13 |
| `gayrimenkul` | Gayrimenkul | 3 | 10 |
| `otomotiv_mobilite` | Otomotiv ve Mobilite | 4 | 11 |
| `sanayi_uretim_tarim` | Sanayi, Üretim ve Tarım | 4 | 15 |
| `kisisel_verimlilik` | Kişisel Verimlilik ve Ofis | 4 | 12 |
| `medya_eglence` | Medya ve Eğlence | 4 | 10 |
| `kriz_destek` | Kriz ve Psikososyal Destek | 3 | 9 |
| `genel_sohbet` | Genel ve Kapsam Dışı | 4 | 8 |

Evrensel sorular (her mesaj için sorulur):

| Ad | Tip | Seçenekler |
|---|---|---|
| `aciliyet` | score | Rutin · Yakın · Yüksek · Kritik |
| `eylem_gerekli` | noul | Somut bir işlem (kayıt, iptal, iade, bloke, araç çağrısı) gerekiyor mu? |
| `insana_devret` | noul | Talep bir insan uzmana devredilmeli mi? |
| `guvenlik` | choice | `guvenli`, `prompt_enjeksiyonu`, `jailbreak_rol_yapma`, `zararli_veya_yasadisi`, `yetkisiz_islem_talebi`, `spam_anlamsiz` |
| `duygu` | choice | `notr`, `memnun`, `endiseli`, `uzgun`, `hayal_kirikligi`, `ofkeli`, `caresiz_kriz` |
| `kisisel_veri` | noul | Kişisel veya hassas veri içeriyor mu? |

Tüm alt alanlar ve niyetlerle birlikte taksonominin tamamı
[`pusula/data/taxonomy.yaml`](pusula/data/taxonomy.yaml) dosyasında ve `GET /v1/taxonomy` adresinde sunulmaktadır.
Bu sorular özel isteklerde de yeniden kullanılabilir:

```python
from pusula import load_taxonomy
tax = load_taxonomy()
sorular = {"alan": tax.domain_question(), **tax.universal, "benim_sorum": {...}}
```

## İyi soru yazmak

- **Her seçenek açıklanmalıdır.** Model açıklamaları okur; `"iade": "Teslim alınan ürünü geri göndermek"` biçimi,
  yalın bir anahtardan belirgin biçimde daha iyi sonuç verir.
- **Seçenekler birbirini dışlamalıdır.** İki seçeneğin örtüştüğü durumlarda sınır açıklamada belirtilmelidir
  (`"Kapsamaz: ..."`).
- **Talimatlar ve seçenekler Türkçe yazılmalıdır;** model bu dilde eğitilmiştir.
- **Soru başına en fazla yaklaşık 20 seçenek önerilir.** Daha büyük etiket kümeleri, `/v1/route`'ta olduğu gibi
  hiyerarşik olarak (önce kategori, ardından kategori içindeki seçenekler) sorulmalıdır.
- **Uygun soru tipi seçilmelidir.** Evet/hayır soruları için `noul`, sıralı seviyeler için `score` (düşükten
  yükseğe), diğer tüm sorular için `choice` kullanılır.
- **Girdinin seçeneklerin dışında kalabileceği durumlarda** bir "diğer" seçeneği eklenmelidir
  (`"diger": "Yukarıdakilerin hiçbiri"`).

## Güven değerini kullanmak

- `choice` / `score`: `confidence`, seçilen seçeneğin olasılığıdır.
- `noul`: cevabın kendisi bir olasılıktır; eşik uygulama tarafından belirlenir (ör. `> 0.5`, yüksek riskli
  işlemlerde `> 0.8`).
- Yaygın bir yaklaşımda yüksek eşiğin üstünde işlem otomatik olarak yapılır, orta bantta onay istenir ve eşiğin
  altında talep bir insana devredilir. Eşikler, uygulamanın kendi trafiğinden alınan bir örneklem üzerinde
  ayarlanmalıdır.

## Örnekler

- [`examples/agentic_router.py`](examples/agentic_router.py): Bir ajan için güvenlik denetimi, araç seçimi, onay ve
  aciliyet kararları tek çağrıda alınır.

## Kendi modelinizi eğitmek

Pusula'nın oluşturulmasında kullanılan tüm bileşenler bu depoda yer almaktadır; bu sayede Decision 3.0 başka
alanların ya da dillerin verisiyle ince ayarlanabilir. Pipeline, yerel bir LLM ile etiketli karar verisi üretir,
bu veriyi temizler ve d3'ü QLoRA ile ince ayarlar:

```bash
pip install -e ".[train,generate,d3]"
bash scripts/pipeline.sh benim-kosum
# generate → judge → fetch_d3 → teach → build → curate → gold → d3train → evaluate
```

Her aşama kaldığı yerden sürdürülebilir. Seçenekler, [`pusula/`](pusula/) altındaki modül açıklamalarında belgelenmiştir.

Takdir gerektiren sorular (aciliyet, insana devretme, eylem gerekliliği) yazılı bir
[etiketleme politikasına](pusula/data/label_policy.md) uyar; politika belirli bir ürüne göre uyarlanabilir. Politika
değiştiğinde `python -m pusula.judge --policy --refresh --questions …` bu soruları açık ağırlıklı yargıç modelle
yeniden etiketler; `python -m pusula.relabel` ise yeni etiketleri altın setle ölçer ve eğitim verisine işler.
`python -m pusula.annotate` (`pip install -e ".[annotate]"`, Claude API) iki etiketçi ve hakemle kör altın set ve
benchmark etiketleri üretir. Ticari modellerin çoğunun kullanım şartları, çıktılarıyla rakip model eğitilmesine izin
vermediğinden bu etiketler yalnızca değerlendirme amacıyla kullanılmalıdır.

## Katkıda bulunanlar ve atıf

Pusula, [Hasan Şahin](https://github.com/hsnsahinhsn) (proje yürütücüsü) tarafından geliştirilmektedir. Projeye
[Ahmet Sayman](https://github.com/asayman) ([Hugging Face](https://huggingface.co/asaymantr)) tarafından önemli
katkılar sağlanmıştır.

Pusula'nın bir araştırmada veya üründe kullanılması durumunda aşağıdaki atıf rica edilir. Aynı bilgi
[`CITATION.cff`](CITATION.cff) dosyasında ve GitHub'daki "Cite this repository" düğmesinde de yer almaktadır.

```bibtex
@software{pusula2026,
  title   = {Pusula: The First Full-Featured Turkish Decision Model},
  author  = {Şahin, Hasan and Sayman, Ahmet},
  year    = {2026},
  version = {1.0},
  url     = {https://github.com/hsnsahinhsn/pusula},
  license = {Apache-2.0}
}
```

## Lisans ve teşekkürler

Pusula, Apache 2.0 lisansıyla yayımlanmaktadır ve şu açık kaynak çalışmalar üzerine kurulmuştur:
[TabiBERT](https://huggingface.co/boun-tabilab/TabiBERT) (Boğaziçi Üniversitesi TABILAB),
[vllm-sr/d3](https://huggingface.co/vllm-sr/d3) (vLLM Semantic Router) ve
[Qwen3.8](https://huggingface.co/Qwen/Qwen3.8-27B). Bu çalışmaların tümü Apache 2.0 lisanslıdır.
`pusula/d3model.py` içindeki d3 prompt sözleşmesi ve nedensel olmayan dikkat kancası, vllm-sr/d3'teki uygulamayı
(orada perplexity-ai/pplx-decider-v1.1-27b'den uyarlanmıştır, Apache 2.0) izlemektedir.
