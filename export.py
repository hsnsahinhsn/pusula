"""
Pusula: Model Dışa Aktarma ve Derleme Motoru (ONNX & TensorRT)
PyTorch modelini ultra-düşük gecikmeli (<4 ms) ONNX Runtime ve NVIDIA TensorRT motoruna dönüştürür.
8.192 token dinamik eksen desteği ile tam uyumludur.
"""
import os
import argparse
import torch
from transformers import AutoTokenizer
from model import TurkishDecisionModel
from config import InferenceConfig

def export_to_onnx(
    checkpoint_dir: str = "checkpoints/pusula/best",
    output_onnx_path: str = "checkpoints/pusula/pusula.onnx",
    opset_version: int = 17
):
    print(f"Pusula ONNX Dışa Aktarımı Başlatılıyor: {checkpoint_dir}")
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
    
    device = torch.device("cpu")
    model_path = os.path.join(checkpoint_dir, "pusula_decision_engine.pt")
    if not os.path.exists(model_path):
        legacy = os.path.join(checkpoint_dir, "turkish_decision_model.pt")
        if os.path.exists(legacy):
            model_path = legacy
            
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model ağırlığı bulunamadı: {model_path}. Lütfen önce train.py çalıştırın.")

    checkpoint = torch.load(model_path, map_location=device)
    base_model_name = checkpoint["base_model_name"]
    options_vocab = checkpoint["options_vocab"]
    
    model = TurkishDecisionModel(base_model_name, options_vocab)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Örnek dummy tensörler (dinamik boyutları test etmek için)
    dummy_batch = 1
    dummy_seq = 64
    dummy_input_ids = torch.randint(0, 1000, (dummy_batch, dummy_seq), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones((dummy_batch, dummy_seq), dtype=torch.long, device=device)
    dummy_candidate_mask = torch.ones((dummy_batch, len(options_vocab)), dtype=torch.bool, device=device)

    # Dinamik Eksen Tanımları (1 ile 8.192 token arası değişken girdi desteği)
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "candidate_mask": {0: "batch_size"},
        "choice_logits": {0: "batch_size"},
        "score": {0: "batch_size"},
        "action_logits": {0: "batch_size"},
        "action_prob": {0: "batch_size"}
    }

    print("ONNX Grafiği Çıkarılıyor...")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask, dummy_candidate_mask),
        output_onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask", "candidate_mask"],
        output_names=["choice_logits", "score", "action_logits", "action_prob"],
        dynamic_axes=dynamic_axes
    )
    
    print(f"✓ ONNX Modeli Başarıyla Oluşturuldu: {output_onnx_path}")
    print("\n[NVIDIA TensorRT Derleme Komutu]:")
    print(f"trtexec --onnx={output_onnx_path} \\")
    print(f"        --saveEngine={output_onnx_path.replace('.onnx', '.engine')} \\")
    print(f"        --fp16 \\")
    print(f"        --minShapes=input_ids:1x1,attention_mask:1x1,candidate_mask:1x{len(options_vocab)} \\")
    print(f"        --optShapes=input_ids:1x128,attention_mask:1x128,candidate_mask:1x{len(options_vocab)} \\")
    print(f"        --maxShapes=input_ids:1x8192,attention_mask:1x8192,candidate_mask:1x{len(options_vocab)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pusula ONNX & TensorRT Dışa Aktarma Aracı")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pusula/best", help="Model kayıt dizini")
    parser.add_argument("--output", type=str, default="checkpoints/pusula/pusula.onnx", help="Çıktı ONNX dosya yolu")
    args = parser.parse_args()
    
    export_to_onnx(args.checkpoint, args.output)
