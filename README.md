# Türkçe Karar Modeli (Laya) Eğitimi & Sentetik Veri Üretim Pipeline'ı

Bu depo, **NVIDIA RTX 4090 (24 GB VRAM)** donanımı üzerinde agentic iş akışlarında, akıllı yönlendirme katmanlarında (routing) ve araç seçiminde (tool calling) kullanılmak üzere optimize edilmiş **düşük gecikmeli (~10-20 ms) Türkçe Tipli Karar Modeli (Laya)** eğitim ve sentetik veri üretim altyapısını içerir.

---

## 🎯 Proje Hedefi & Mimari

Geleneksel üretken LLM'leri (GPT-4, Claude, Qwen) her karar veya yönlendirme adımında çalıştırmak yüksek gecikme (500 ms - 2000 ms) ve yüksek maliyet yaratır. 

**Laya Mimarisi:**
- **Omurga (Backbone):** ModernBERT-large (421M parametre) veya `convaiinnovations/laya-multilingual` (mmBERT-base, 100+ dil Türkçe dahil).
- **Tipli Karar Başlığı (Typed Decision Head):** Tek bir forward pass içinde şu 3 çıktıyı üretir:
  1. `choice`: Aday seçenekler (`options`) arasından en uygun kategorik karar.
  2. `score`: 0.0 - 1.0 arası kalibre edilmiş güven / öncelik / risk skoru.
  3. `action_required`: Boolean (true/false) eylem veya onay filtresi.
- **Hız:** RTX 4090 üzerinde **<15 ms** çıkarım gecikmesi.

---

## 📁 Proje Yapısı

```
turkish_laya_pipeline/
├── config.py                  # Dikey alanlar, model ve eğitim hiperparametreleri
├── synthetic_generator.py     # Yerel Qwen LLM ile sentetik Türkçe JSONL veri üretici
├── model.py                   # PyTorch Laya Decision Model & Typed Decision Head
├── train.py                   # bfloat16 + Flash Attention eğitim ve kalibrasyon döngüsü
├── serve.py                   # Düşük gecikmeli FastAPI mikroservisi
├── agentic_router_example.py  # Agentic iş akışında kullanım örneği
├── requirements.txt           # Gerekli kütüphaneler
└── data/                      # train.jsonl, val.jsonl, test.jsonl
```

---

## 🚀 RTX 4090 Kurulum ve Adım Adım Çalıştırma Rehberi

### 1. Adım: Sanal Ortam ve Kütüphaneler

```bash
python -m venv venv
# Linux / WSL:
source venv/bin/activate
# Windows:
.\venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Adım: Yerel LLM Sunucusunun Başlatılması (Sentetik Veri İçin)

Aynı RTX 4090 üzerinde **Qwen 2.5 7B** veya **14B** modelini vLLM ile başlatın (yaklaşık 12-14 GB VRAM kullanır):

```bash
# vLLM ile OpenAI uyumlu yerel sunucu:
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.6 \
  --max-model-len 4096
```

*(Alternatif olarak Ollama kullanıyorsanız: `ollama run qwen2.5:7b`)*

### 3. Adım: Sentetik Türkçe Veri Üretimi

4 dikey alanda (Tool Calling, Müşteri Hizmetleri, Güvenlik Guardrails, Aciliyet Skorlama) doğrulanmış JSONL verisi üretmek için:

```bash
python synthetic_generator.py
```
Bu script üretilen her veriyi Pydantic ile şema kontrolünden geçirir, kopyaları eler ve `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl` dosyalarını oluşturur.

### 4. Adım: Model Eğitimi & Sıcaklık Kalibrasyonu (RLCD)

Eğitimi başlatmadan önce vLLM sürecini durdurarak 24 GB VRAM'in tamamını eğitime ayırabilirsiniz (Model sadece 4-6 GB VRAM tüketir):

```bash
python train.py
```
- RTX 4090'ın `bfloat16` ve `FlashAttention-2` yeteneklerini tam kapasite kullanır.
- Eğitim bitiminde **Platt Scaling (Sıcaklık Kalibrasyonu)** yaparak aşırı güveni (overconfidence) engeller ve Brier Skorunu optimize eder.
- En iyi ağırlıklar `checkpoints/laya_turkish_decision/best/` dizinine kaydedilir.

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
       "options": ["muhasebe_finans", "teknik_destek", "lojistik_kargo"]
     }'
```

Yanıt (<15 ms):
```json
{
  "choice": "muhasebe_finans",
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
