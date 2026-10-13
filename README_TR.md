<div align="center">

# Pusula

### İlk Türkçe tam donanımlı karar modeli

[English](README.md) • [Türkçe](README_TR.md)

[![CI](https://github.com/hsnsahinhsn/pusula/actions/workflows/ci.yml/badge.svg)](https://github.com/hsnsahinhsn/pusula/actions)
[![Lisans: Apache 2.0](https://img.shields.io/badge/Lisans-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)

</div>

Karar modelleri bir metin hakkında tipli sorulara cevap verir (bu kaydı kime yönlendirmeli, bu bir
prompt enjeksiyonu mu, ne kadar acil) ve metin üretmek yerine **her seçenek için bir olasılık** döndürür.
Bugünkü karar modelleri (Decision 3.0, Jev, Laya, ...) İngilizce için kurulmuştur ve Türkçede zorlanır.
**Pusula, Türkçe için eğitilmiş Decision 3.0'dır.**

- **Önce Türkçe.** [vllm-sr/d3](https://huggingface.co/vllm-sr/d3) (Decision 3.0), 20 alanda Türkçe karar
  verisiyle ince ayarlandı: bankacılık, muhasebe ve vergi, hukuk, sağlık, e-ticaret, lojistik, kamu
  hizmetleri ve daha fazlası. Gündelik yazım, yazım hataları, eksik Türkçe karakterler ve Türkçe–İngilizce
  karışık metinler dahil.
- **İstek anında istediğiniz seçenekler.** Soruyu ve seçeneklerini her istekte siz tanımlarsınız.
- **Üç soru tipi.** `choice` (birini seç), `noul` (evet/hayır), `score` (sıralı seviyeler).
- **d3'ün yerine doğrudan takılır.** d3'ün `/v1/systemone` istek ve cevap biçimiyle birebir aynı.
- **Hazır Türkçe taksonomi.** 20 alan → 86 alt alan → 299 niyet ve altı evrensel kontrol
  (aciliyet, eylem gerekli mi, insana devir, güvenlik riski, duygu, kişisel veri).
- **Türkçe karar benchmark'ı.** Pusula, bağımsız etiketlenmiş Türkçe bir test setinde ham Decision 3.0 ile
  karşılaştırılır; sonuçlar modelle birlikte yayınlanır.

> **Durum:** v1.0. Ağırlıklar: [Pusula](https://huggingface.co/hsnsahinhsn/pusula) ·
> [Pusula Mini](https://huggingface.co/hsnsahinhsn/pusula-mini) · [Pusula Nano](https://huggingface.co/hsnsahinhsn/pusula-nano).

## Modeller

| Model | Boyut | Çalıştığı yer | Benchmark / altın set | Ne için |
|---|---|---|---|---|
| **Pusula** (Türkçe Decision 3.0) | 26B: `vllm-sr/d3` üzerine 0,9 GB'lık LoRA adaptörü | GPU: BF16'da ≥ 64 GB, 4-bit'te ≥ 24 GB | **92,8 / 90,3** | En yüksek doğruluk; hazır sorularda da sizin sorularınızda da |
| **Pusula Mini** | 4,7B (Qwen3.5-4B, tam eğitim, d3 biçimi) | GPU (ağırlıklar BF16'da 9,1 GB) | 80,9 / 84,5 | Daha küçük bir GPU; eğitim verisine benzemeyen görevlerde en güçlü küçük model |
| **Pusula Nano** | 149M ([TabiBERT](https://huggingface.co/boun-tabilab/TabiBERT) cross-encoder) | CPU veya herhangi bir GPU | 81,8 / 84,2 | Hazır sorularda yüksek hacim ve düşük gecikme |

Hepsi birebir aynı istekleri cevaplar.

---

## Sonuçlar

Doğruluk (%). Değerlendirme etiketleri birbirini görmeden çalışan iki etiketçiden gelir (Claude Opus 5.5 ve Claude
Sonnet 5.5); ayrıştıkları yerde hakem karar verir ve hepsi yayınlanan [etiketleme politikasına](pusula/data/label_policy.md)
uyar. Bu etiketler yalnız ölçüm için kullanılır, hiçbir zaman eğitim etiketi olarak kullanılmaz.

| Değerlendirme | Soru | Decision 3.0 | **Pusula** | Pusula Mini | Pusula Nano |
|---|---|---|---|---|---|
| [Türkçe Karar Benchmark'ı](benchmark/README.md) (2.194 mesaj) | 19.861 | 88,5 | **92,8** | 80,9 | 81,8 |
| Altın test seti (eğitimde olmayan 759 mesaj) | 7.015 | 85,5 | **90,3** | 84,5 | 84,2 |
| Kendi sorularınız (497 mesaj × 3 özel soru) | 1.491 | **94,8** | 93,1 | 85,0 | 85,6 |
| MASSIVE tr-TR, 60 seçenekli niyet, sıfır örnek (insan yazımı) | 2.974 | **85,3** | 84,7 | 33,3 | 30,9 |
| OffensEval 2020 Türkçe, saldırgan dil (gerçek tweet'ler) | 3.528 | 88,4 | **88,8** | 76,9 | 27,5 |

Altın test seti, soru bazında:

| Soru | Decision 3.0 | **Pusula** |
|---|---|---|
| Aciliyet (4 seviye) | 53,2 | **80,2** |
| İnsana devretme | 81,5 | **89,1** |
| Alan | 82,1 | **87,4** |
| Duygu | 85,0 | **88,9** |
| Güvenlik riski | 94,2 | **97,8** |
| Eylem gerekli mi | 88,9 | **89,2** |
| Kişisel veri | **96,4** | 96,0 |
| Niyet | **96,8** | 95,1 |

- **Aciliyet kısmen öznel:** iki etiketçi mesajların %84,4'ünde aynı cevabı veriyor; 80,2 bu tavana yakın.
- **Kendi sorularınız:** Pusula genel olarak Decision 3.0 ile aynı seviyede (altın setteki özel sorularda +2,2,
  OffensEval'de +0,4, MASSIVE'de −0,6), ancak 10 ve üzeri seçenekli sorularda geride (86,7'ye 90,1). Uzun seçenek
  listeleri için bir güncelleme planlandı.
- **Mini ve Nano** ağırlıkla hazır sorularla eğitildi. Bunlara benzemeyen, çok seçenekli görevler için Pusula'yı
  kullanın; küçük modeller arasında bu tür görevlerde Mini çok daha iyi (OffensEval'de 76,9'a 27,5).
- MASSIVE ve OffensEval eğitim verisinde yok.

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
- [Lisans ve teşekkürler](#lisans-ve-teşekkürler)

---

## Kurulum

```bash
pip install "pusula[serve,d3] @ git+https://github.com/hsnsahinhsn/pusula"
hf download vllm-sr/d3 --local-dir ~/d3 --exclude "assets/*"     # temel Decision 3.0 ağırlıkları (~52 GB)
hf download hsnsahinhsn/pusula --local-dir ~/pusula                # Pusula adaptörü (~0,9 GB)
# ya da temel ağırlık gerektirmeyen küçük modellerden biri:
hf download hsnsahinhsn/pusula-mini --local-dir ~/pusula-mini      # Pusula Mini (~9,1 GB)
hf download hsnsahinhsn/pusula-nano --local-dir ~/pusula-nano      # Pusula Nano (~0,6 GB)
```

`serve` eki PyTorch, Transformers ve FastAPI'yi; `d3` eki 4-bit model için PEFT ve bitsandbytes'ı kurar.
Önce PyTorch'un CUDA sürümünü kurun ([pytorch.org](https://pytorch.org/get-started/locally/)).
Pusula Mini ve Pusula Nano için yalnızca `serve` eki yeterlidir.

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

Yukarıdaki isteğin cevabı (değerler örnek amaçlıdır):

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
# engine = D3Engine.load("~/pusula-mini")                # Pusula Mini (d3 biçimi, adaptörsüz)
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

Birden çok isteği tek seferde (mümkün olan en az ileri geçişle) çalıştırmak için:

```python
results = engine.decide_many([(metin, sorular) for metin in mesajlar])
```

Sorular yukarıdaki gibi sözlük ya da `pusula.Question` nesnesi olabilir.

## HTTP API

Sunucuyu başlatın:

```bash
PUSULA_D3_BASE=~/d3 PUSULA_MODEL_DIR=path/to/pusula pusula-serve --port 8080   # Pusula
PUSULA_D3_BASE=~/pusula-mini pusula-serve --port 8080                         # Pusula Mini
PUSULA_MODEL_DIR=path/to/pusula-nano pusula-serve --port 8080                  # Pusula Nano
```

Sunucu varsayılan olarak yalnız `127.0.0.1` üzerinde dinler. Başka makinelerden erişmek için önce bir API anahtarı
belirleyip `--host 0.0.0.0` verin; istemciler her `/v1/*` isteğinde `Authorization: Bearer <anahtar>` (ya da
`X-API-Key: <anahtar>`) gönderir:

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

Geçersiz sorular HTTP 422 ve sebebiyle döner. Ayarlar (ortam değişkenleri):

| Değişken | Varsayılan | Anlamı |
|---|---|---|
| `PUSULA_D3_BASE` | — | Temel `vllm-sr/d3` dizini; Pusula'yı sunmak için ayarlayın (`PUSULA_MODEL_DIR` adaptör olur) |
| `PUSULA_MODEL_DIR` | — | Pusula adaptör dizini ya da Pusula Nano model dizini |
| `PUSULA_DEVICE` | otomatik | `cuda`, `cuda:1`, `cpu`… |
| `PUSULA_CORS_ORIGINS` | yok | Tarayıcı istemcileri için virgülle ayrılmış izinli origin'ler |
| `PUSULA_MAX_QUESTIONS` | 32 | İstek başına soru sayısı |
| `PUSULA_MAX_BATCH` | 64 | Toplu çağrıdaki istek sayısı |
| `PUSULA_MAX_STATE_CHARS` | 60000 | En uzun durum metni |

## Hazır taksonomi ve yönlendirme

`POST /v1/route` (ya da `engine.route(metin)`) mesajı üç adımda yönlendirir: **alan** →
**alt_alan** → **niyet**. Evrensel soruları da aynı anda cevaplar:

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

Evrensel sorular (her mesaja sorulur):

| Ad | Tip | Seçenekler |
|---|---|---|
| `aciliyet` | score | Rutin · Yakın · Yüksek · Kritik |
| `eylem_gerekli` | noul | Somut bir işlem (kayıt, iptal, iade, bloke, araç çağrısı) gerekiyor mu? |
| `insana_devret` | noul | Bir insan uzman devralmalı mı? |
| `guvenlik` | choice | `guvenli`, `prompt_enjeksiyonu`, `jailbreak_rol_yapma`, `zararli_veya_yasadisi`, `yetkisiz_islem_talebi`, `spam_anlamsiz` |
| `duygu` | choice | `notr`, `memnun`, `endiseli`, `uzgun`, `hayal_kirikligi`, `ofkeli`, `caresiz_kriz` |
| `kisisel_veri` | noul | Kişisel veya hassas veri içeriyor mu? |

Tüm alt alanlar ve niyetlerle birlikte taksonominin tamamı
[`pusula/data/taxonomy.yaml`](pusula/data/taxonomy.yaml) dosyasında ve `GET /v1/taxonomy` adresinde.
Bu soruları kendi isteklerinizde de kullanabilirsiniz:

```python
from pusula import load_taxonomy
tax = load_taxonomy()
sorular = {"alan": tax.domain_question(), **tax.universal, "benim_sorum": {...}}
```

## İyi soru yazmak

- **Her seçeneği açıklayın.** Model açıklamayı okur; `"iade": "Teslim alınan ürünü geri göndermek"`
  çıplak bir anahtardan çok daha iyi çalışır.
- **Seçenekler birbirini dışlasın.** İki seçenek örtüşüyorsa sınırı açıklamada belirtin (`"Kapsamaz: ..."`).
- **Talimat ve seçenekleri Türkçe yazın.** Model bunun üzerine eğitilmiştir.
- **Soru başına en fazla ~20 seçenek tercih edin.** Daha büyük etiket kümelerinde, `/v1/route`'un yaptığı
  gibi hiyerarşik sorun (önce kategori, sonra kategorinin içindeki seçenekler).
- **Doğru tipi seçin.** Evet/hayır → `noul`; sıralı seviyeler → `score` (düşükten yükseğe); diğer her şey → `choice`.
- **Gerekirse bir "diğer" seçeneği ekleyin** (`"diger": "Yukarıdakilerin hiçbiri"`).

## Güven değerini kullanmak

- `choice` / `score`: `confidence`, seçilen seçeneğin olasılığıdır.
- `noul`: cevabın kendisi bir olasılıktır; eşiği siz seçin (ör. `> 0.5`, riskli işlemlerde `> 0.8`).
- Sık kullanılan desen: yüksek eşiğin üstünde otomatik işlem, orta bantta onay isteme, altında insana devir.
  Eşikleri kendi trafiğinizden bir örneklem üzerinde ayarlayın.

## Örnekler

- [`examples/agentic_router.py`](examples/agentic_router.py): Bir ajan için güvenlik denetimi, araç seçimi,
  onay ve aciliyet kararlarını tek çağrıda verir.

## Kendi modelinizi eğitmek

Pusula'yı oluşturmak için kullanılan her şey bu depoda; Decision 3.0'ı kendi alanınızın ya da dilinizin
verisiyle ince ayarlayabilirsiniz. Pipeline, yerel bir LLM ile etiketli karar verisi üretir, temizler ve
d3'ü QLoRA ile ince ayarlar:

```bash
pip install -e ".[train,generate,d3]"
bash scripts/pipeline.sh benim-kosum
# generate → judge → fetch_d3 → teach → build → curate → gold → d3train → evaluate
```

Her aşama kaldığı yerden devam eder. Seçenekler için [`pusula/`](pusula/) altındaki modül açıklamalarına bakın.

Takdir gerektiren sorular (aciliyet, insana devretme, eylem gerekliliği) yazılı bir
[etiketleme politikasına](pusula/data/label_policy.md) uyar; kendi ürününüze göre düzenleyebilirsiniz. Politika
değişince `python -m pusula.judge --policy --refresh --questions …` bu soruları açık ağırlıklı yargıç modelle yeniden
etiketler, `python -m pusula.relabel` yeni etiketleri altın setinizle ölçer ve eğitim verisine işler.
`python -m pusula.annotate` (`pip install -e ".[annotate]"`, Claude API) iki etiketçi ve hakemle kör altın set ve
benchmark etiketleri üretir. Bu etiketleri yalnızca değerlendirmede kullanın: ticari model şartlarının çoğu, çıktılarıyla
rakip model eğitmeye izin vermez.

## Lisans ve teşekkürler

Apache 2.0. Pusula şu açık kaynak çalışmalar üzerine kuruludur:
[TabiBERT](https://huggingface.co/boun-tabilab/TabiBERT) (Boğaziçi Üniversitesi TABILAB),
[vllm-sr/d3](https://huggingface.co/vllm-sr/d3) (vLLM Semantic Router) ve
[Qwen3.8](https://huggingface.co/Qwen/Qwen3.8-27B). Hepsi Apache 2.0 lisanslıdır. `pusula/d3model.py`
içindeki d3 prompt sözleşmesi ve nedensel olmayan dikkat kancası vllm-sr/d3'ü (orada
perplexity-ai/pplx-decider-v1.1-27b'den uyarlanmış) izler, Apache 2.0.
