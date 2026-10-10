"""
Türkçe Tipli Karar Modeli (Turkish Typed Decision Model) Mimarisi
Modern multi-task encoder omurgası üzerine kurulu tek geçişli (single-pass) tipli karar başlığı.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel
from typing import Dict, List, Optional, Tuple

class DecisionHead(nn.Module):
    """Tipli Karar Başlığı (Typed Decision Head)
    
    Tek bir encoder geçişinden çıkan CLS/Mean pooling tensörünü alarak aynı anda:
    1. choice_logits: Aktif aday seçenekler üzerinde kategorik dağılım
    2. score: 0.0 - 1.0 arası öncelik/güven skoru (Sigmoid çıkışlı)
    3. action_logits: boolean onay/müdahale filtresi (BCE logits)
    üretir.
    """
    def __init__(self, hidden_size: int, num_total_options: int, dropout: float = 0.1):
        super().__init__()
        # Ortak Projeksiyon ve Normalizasyon Katmanı
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # 1. Choice Head (Kategorik Seçim)
        self.choice_classifier = nn.Linear(hidden_size, num_total_options)
        
        # 2. Score Head (Regresyon 0.0 - 1.0)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # 3. Action Required Head (Boolean Logits)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, pooled_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.dropout(self.activation(self.layer_norm(self.dense(pooled_features))))
        
        choice_logits = self.choice_classifier(features)
        score_preds = self.score_head(features).squeeze(-1)       # [batch_size]
        action_logits = self.action_head(features).squeeze(-1)   # [batch_size]
        
        return choice_logits, score_preds, action_logits


class TurkishDecisionModel(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        options_vocabulary: List[str],
        dropout: float = 0.1
    ):
        super().__init__()
        self.base_model_name = base_model_name
        self.options_vocab = list(options_vocabulary)
        self.option_to_id = {opt: idx for idx, opt in enumerate(self.options_vocab)}
        self.id_to_option = {idx: opt for idx, opt in enumerate(self.options_vocab)}
        self.num_options = len(self.options_vocab)
        
        # Encoder Omurgasını Yükle
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(base_model_name, config=self.config)
        hidden_size = getattr(self.config, "hidden_size", 768)
        
        # Pusula Tipli Karar Başlığı
        self.decision_head = DecisionHead(
            hidden_size=hidden_size,
            num_total_options=self.num_options,
            dropout=dropout
        )
        
        # Kalibrasyon Sıcaklık Değerleri (Platt/Temperature Scaling)
        self.temperature_choice = nn.Parameter(torch.ones(1), requires_grad=False)
        self.temperature_action = nn.Parameter(torch.ones(1), requires_grad=False)

    def _pool(self, model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        """CLS veya Mean Pooling hesaplar."""
        last_hidden_state = model_output.last_hidden_state
        # Eğer model CLS token'a sahipse CLS kullanılır, yoksa maskeli ortalama alınır
        if hasattr(model_output, "pooler_output") and model_output.pooler_output is not None:
            return model_output.pooler_output
        # ModernBERT için CLS token (0. indeks)
        return last_hidden_state[:, 0, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
        candidate_mask: [batch_size, num_options] - Yalnızca soruda verilen seçenekler için 1.0, diğerleri için 0.0
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(outputs, attention_mask)
        
        choice_logits, score_preds, action_logits = self.decision_head(pooled)
        
        # Eğer aday seçenek maskesi verilmişse, aday olmayan seçeneklerin logitlerini -inf yap
        if candidate_mask is not None:
            masked_logits = choice_logits.masked_fill(~candidate_mask.bool(), -1e9)
        else:
            masked_logits = choice_logits
            
        return {
            "choice_logits": masked_logits,
            "raw_choice_logits": choice_logits,
            "score": score_preds,
            "action_logits": action_logits,
            "action_prob": torch.sigmoid(action_logits / self.temperature_action)
        }

    def predict_single(
        self,
        tokenizer,
        context: str,
        options: List[str],
        device: torch.device,
        max_length: int = 8192
    ) -> Dict[str, any]:
        """Tek bir istek için dinamik uzunlukla (<8192 token) çıkarım yapar."""
        self.eval()
        # Bağlamı ve seçenekleri formatla
        prompt = f"Bağlam: {context}\nSeçenekler: {', '.join(options)}"
        inputs = tokenizer(
            prompt,
            max_length=max_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Aday seçenek maskesini hazırla
        candidate_mask = torch.zeros((1, self.num_options), dtype=torch.bool, device=device)
        valid_indices = []
        for opt in options:
            if opt in self.option_to_id:
                idx = self.option_to_id[opt]
                candidate_mask[0, idx] = True
                valid_indices.append(idx)
                
        with torch.inference_mode():
            outputs = self.forward(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                candidate_mask=candidate_mask if valid_indices else None
            )
            
            # Sıcaklık ölçeklendirmeli Softmax
            scaled_logits = outputs["choice_logits"] / self.temperature_choice
            probs = torch.softmax(scaled_logits, dim=-1)[0]
            
            best_idx = torch.argmax(scaled_logits[0]).item()
            best_choice = self.id_to_option.get(best_idx, options[0] if options else "unknown")
            choice_confidence = probs[best_idx].item()
            
            score_val = outputs["score"][0].item()
            action_prob = outputs["action_prob"][0].item()
            action_required = action_prob >= 0.5
            
        return {
            "choice": best_choice,
            "choice_confidence": round(choice_confidence, 4),
            "score": round(score_val, 4),
            "action_required": action_required,
            "action_probability": round(action_prob, 4)
        }

# Pusula & Geriye dönük uyumluluk alias'ları
Pusula = TurkishDecisionModel
PusulaDecisionEngine = TurkishDecisionModel
TurkishLayaDecisionModel = TurkishDecisionModel
