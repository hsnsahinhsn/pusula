"""
Pusula: Turkish Decision Engine (FastAPI)
NVIDIA GPU üzerinde tek geçişli tipli çıkarım ve triyaj mikroservisi (<15 ms).
Web, Mobil ve Ajanik istemciler için kurumsal REST API.
"""
import os
import time
from typing import List, Optional, Dict
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from model import TurkishDecisionModel, PusulaDecisionEngine
from config import GenerationConfig

# API İstek ve Yanıt Şemaları
class DecisionRequest(BaseModel):
    context: str = Field(..., description="Kullanıcı bildirimi, bilet metni veya ajan bağlamı", min_length=2)
    options: List[str] = Field(..., description="Aday karar seçenekleri listesi", min_length=2)

class BatchDecisionRequest(BaseModel):
    items: List[DecisionRequest] = Field(..., description="Toplu karar istekleri listesi")

class DecisionResponse(BaseModel):
    choice: str = Field(..., description="Seçilen karar etiketi")
    choice_confidence: float = Field(..., description="Kalibre edilmiş seçim güven olasılığı (0.0 - 1.0)")
    score: float = Field(..., description="Öncelik / güven / risk skoru (0.0 - 1.0)")
    action_required: bool = Field(..., description="Müdahale / harici işlem gerekiyor mu?")
    latency_ms: float = Field(..., description="Model çıkarım gecikmesi (milisaniye)")

class BatchDecisionResponse(BaseModel):
    results: List[DecisionResponse]
    total_latency_ms: float

class TaxonomyResponse(BaseModel):
    total_domains: int
    domains: Dict[str, Dict]

app = FastAPI(
    title="Pusula: Turkish Decision Engine API",
    description="Sub-15 ms typed decision and triage engine for autonomous Turkish agent workflows, web dashboards, and mobile clients.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Web ve Mobil istemciler için Cross-Origin (CORS) Desteği
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Model & Tokenizer Değişkenleri
MODEL: Optional[TurkishDecisionModel] = None
TOKENIZER = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GEN_CONFIG = GenerationConfig()

@app.on_event("startup")
def load_decision_engine():
    global MODEL, TOKENIZER
    checkpoint_dir = os.getenv("CHECKPOINT_DIR", "checkpoints/pusula/best")
    if not os.path.exists(checkpoint_dir):
        fallback_dir = "checkpoints/turkish_decision_model/best"
        if os.path.exists(fallback_dir):
            checkpoint_dir = fallback_dir

    model_path = os.path.join(checkpoint_dir, "pusula_decision_engine.pt")
    if not os.path.exists(model_path):
        legacy_tdm = os.path.join(checkpoint_dir, "turkish_decision_model.pt")
        if os.path.exists(legacy_tdm):
            model_path = legacy_tdm
        else:
            legacy_laya = os.path.join(checkpoint_dir, "laya_turkish_model.pt")
            if os.path.exists(legacy_laya):
                model_path = legacy_laya

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        options_vocab = checkpoint["options_vocab"]
        base_model_name = checkpoint["base_model_name"]
        
        TOKENIZER = AutoTokenizer.from_pretrained(checkpoint_dir)
        MODEL = TurkishDecisionModel(
            base_model_name=base_model_name,
            options_vocabulary=options_vocab
        )
        MODEL.load_state_dict(checkpoint["model_state_dict"])
        MODEL.to(DEVICE)
        MODEL.eval()
        
        if torch.cuda.is_available() and hasattr(torch, "compile"):
            try:
                MODEL = torch.compile(MODEL, mode="reduce-overhead")
            except Exception:
                pass
                
        print(f"✓ Model başarıyla yüklendi: {checkpoint_dir} on {DEVICE}")
    else:
        print(f"Bilgi: Model dosyası bulunamadı ({model_path}). Geliştirme/Mock modu devrede.")

@app.post("/decide", response_model=DecisionResponse, tags=["Inference"])
async def decide(req: DecisionRequest):
    """Tek bir bağlam için anlık tipli karar ve triyaj üretir (<15 ms)."""
    start_time = time.perf_counter()
    
    if MODEL is None or TOKENIZER is None:
        latency = (time.perf_counter() - start_time) * 1000
        return DecisionResponse(
            choice=req.options[0],
            choice_confidence=0.90,
            score=0.85,
            action_required=True,
            latency_ms=round(latency, 2)
        )
        
    try:
        result = MODEL.predict_single(
            tokenizer=TOKENIZER,
            context=req.context,
            options=req.options,
            device=DEVICE
        )
        latency = (time.perf_counter() - start_time) * 1000
        
        return DecisionResponse(
            choice=result["choice"],
            choice_confidence=result["choice_confidence"],
            score=result["score"],
            action_required=result["action_required"],
            latency_ms=round(latency, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-decide", response_model=BatchDecisionResponse, tags=["Inference"])
async def batch_decide(req: BatchDecisionRequest):
    """Çoklu istekleri tek seferde işler (Web dashboard ve toplu iş akışları için)."""
    t0 = time.perf_counter()
    results = []
    for item in req.items:
        res = await decide(item)
        results.append(res)
    total_latency = (time.perf_counter() - t0) * 1000
    return BatchDecisionResponse(results=results, total_latency_ms=round(total_latency, 2))

@app.get("/taxonomies", response_model=TaxonomyResponse, tags=["Metadata"])
def get_taxonomies():
    """Web ve mobil arayüzler için yerleşik makro-alanları ve seçenekleri döner."""
    return TaxonomyResponse(
        total_domains=len(GEN_CONFIG.taxonomies),
        domains=GEN_CONFIG.taxonomies
    )

@app.get("/health", tags=["Monitoring"])
def health_check():
    """Kubernetes liveness/readiness ve yük dengeleyici sağlık kontrolü."""
    return {
        "status": "healthy",
        "service": "pusula-decision-engine",
        "version": "1.0.0",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": MODEL is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8080, reload=False)
