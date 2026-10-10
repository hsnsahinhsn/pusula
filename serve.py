"""
Pusula: Turkish Decision Engine (FastAPI)
NVIDIA GPU üzerinde tek geçişli tipli çıkarım ve triyaj mikrosanite servisi (<15 ms).
"""
import os
import time
from typing import List, Optional
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from model import TurkishDecisionModel, PusulaDecisionEngine

# API İstek ve Yanıt Şemaları
class DecisionRequest(BaseModel):
    context: str = Field(..., description="Kullanıcı bildirimi, bilet metni veya ajan bağlamı", min_length=3)
    options: List[str] = Field(..., description="Aday karar seçenekleri listesi", min_length=2)

class BatchDecisionRequest(BaseModel):
    items: List[DecisionRequest]

class DecisionResponse(BaseModel):
    choice: str = Field(..., description="Seçilen karar etiketi")
    choice_confidence: float = Field(..., description="Kalibre edilmiş seçim güven olasılığı (0.0 - 1.0)")
    score: float = Field(..., description="Öncelik / güven / risk skoru (0.0 - 1.0)")
    action_required: bool = Field(..., description="Müdahale / harici işlem gerekiyor mu?")
    latency_ms: float = Field(..., description="Model çıkarım gecikmesi (milisaniye)")

app = FastAPI(
    title="Pusula: Turkish Decision Engine API",
    description="Sub-15 ms typed decision and triage engine for autonomous Turkish agent workflows.",
    version="1.0.0"
)

# Global Model & Tokenizer Değişkenleri
MODEL: Optional[TurkishDecisionModel] = None
TOKENIZER = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
        # RTX 4090 için torch.compile hızlandırması (opsiyonel)
        if torch.cuda.is_available() and hasattr(torch, "compile"):
            try:
                MODEL = torch.compile(MODEL, mode="reduce-overhead")
            except Exception:
                pass
                
        print(f"✓ Model başarıyla yüklendi: {checkpoint_dir} on {DEVICE}")
    else:
        print(f"Uyarı: Model dosyası bulunamadı ({model_path}). Mock mod devrede.")

@app.post("/decide", response_model=DecisionResponse)
async def decide(req: DecisionRequest):
    start_time = time.perf_counter()
    
    if MODEL is None or TOKENIZER is None:
        # Model henüz eğitilmediyse veya mock moddaysa güvenli geri dönüş
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

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": MODEL is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8080, reload=False)
