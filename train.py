"""
Türkçe Laya Tipli Karar Modeli Eğitim & Kalibrasyon Pipeline'ı
NVIDIA RTX 4090 (24 GB VRAM) üzerinde bfloat16 ve Flash Attention desteğiyle çalışır.
"""
import json
import os
import time
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, brier_score_loss
from rich.console import Console
from rich.table import Table

from config import TrainingConfig, GenerationConfig
from model import TurkishLayaDecisionModel

console = Console()

class LayaDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        option_to_id: Dict[str, int],
        max_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.option_to_id = option_to_id
        self.num_options = len(option_to_id)
        self.max_length = max_length
        self.records = []
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.records[idx]
        context = item["context"]
        options = item["options"]
        choice = item["choice"]
        score = float(item["score"])
        action_required = 1.0 if item["action_required"] else 0.0

        # Model girdi biçimi
        text = f"Bağlam: {context}\nSeçenekler: {', '.join(options)}"
        
        # Aday seçenek maskesi (Bu soruda hangi seçenekler mevcuttu?)
        candidate_mask = np.zeros(self.num_options, dtype=np.float32)
        for opt in options:
            if opt in self.option_to_id:
                candidate_mask[self.option_to_id[opt]] = 1.0

        # Hedef indeks
        target_choice_idx = self.option_to_id.get(choice, 0)

        return {
            "text": text,
            "candidate_mask": candidate_mask,
            "target_choice_idx": target_choice_idx,
            "target_score": score,
            "target_action": action_required
        }

def collate_fn(batch: List[Dict[str, Any]], tokenizer, max_length: int) -> Dict[str, torch.Tensor]:
    texts = [item["text"] for item in batch]
    tokenized = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    candidate_masks = torch.tensor(np.array([item["candidate_mask"] for item in batch]), dtype=torch.bool)
    choice_targets = torch.tensor([item["target_choice_idx"] for item in batch], dtype=torch.long)
    score_targets = torch.tensor([item["target_score"] for item in batch], dtype=torch.float32)
    action_targets = torch.tensor([item["target_action"] for item in batch], dtype=torch.float32)

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "candidate_mask": candidate_masks,
        "target_choice": choice_targets,
        "target_score": score_targets,
        "target_action": action_targets
    }

class LayaTrainer:
    def __init__(self, train_cfg: TrainingConfig, gen_cfg: GenerationConfig):
        self.cfg = train_cfg
        self.gen_cfg = gen_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Tüm olası seçeneklerin global kelime dağarcığını oluştur
        self.all_options = []
        for d in self.gen_cfg.taxonomies.values():
            self.all_options.extend(d["options"])
        self.all_options = sorted(list(set(self.all_options)))
        self.option_to_id = {opt: idx for idx, opt in enumerate(self.all_options)}
        
        console.print(f"[bold green]Seçenek Havuzu Boyutu: {len(self.all_options)} adet etiket[/bold green]")
        
        # Tokenizer Yükle
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.base_model_name)
        except Exception:
            console.print(f"[yellow]{self.cfg.base_model_name} yüklenemedi, yedek omurga ({self.cfg.fallback_model_name}) deneniyor...[/yellow]")
            self.cfg.base_model_name = self.cfg.fallback_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.base_model_name)
            
        # Model Oluştur
        self.model = TurkishLayaDecisionModel(
            base_model_name=self.cfg.base_model_name,
            options_vocabulary=self.all_options
        ).to(self.device)

        # Kayıp Fonksiyonları
        self.criterion_choice = nn.CrossEntropyLoss()
        self.criterion_score = nn.MSELoss()
        self.criterion_action = nn.BCEWithLogitsLoss()

    def train(self, train_path: str = "data/train.jsonl", val_path: str = "data/val.jsonl"):
        train_dataset = LayaDataset(train_path, self.tokenizer, self.option_to_id, self.cfg.max_length)
        val_dataset = LayaDataset(val_path, self.tokenizer, self.option_to_id, self.cfg.max_length)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, self.tokenizer, self.cfg.max_length)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, self.tokenizer, self.cfg.max_length)
        )

        # Optimizer: Omurga ve Karar Başlığı için Farklı Öğrenme Hızları
        optimizer_grouped_parameters = [
            {"params": self.model.encoder.parameters(), "lr": self.cfg.learning_rate_backbone},
            {"params": self.model.decision_head.parameters(), "lr": self.cfg.learning_rate_heads}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=self.cfg.weight_decay)
        
        total_steps = len(train_loader) * self.cfg.num_train_epochs // self.cfg.gradient_accumulation_steps
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.cfg.warmup_ratio),
            num_training_steps=total_steps
        )

        console.print(f"[bold cyan]RTX 4090 Üzerinde Eğitim Başlıyor (bfloat16)...[/bold cyan]")
        console.print(f"Toplam Adım: {total_steps} | Epochs: {self.cfg.num_train_epochs}\n")

        best_val_acc = 0.0
        scaler = torch.amp.GradScaler('cuda', enabled=(self.cfg.mixed_precision == "fp16"))
        use_bf16 = (self.cfg.mixed_precision == "bf16" and torch.cuda.is_bf16_supported())

        for epoch in range(1, self.cfg.num_train_epochs + 1):
            self.model.train()
            running_loss = 0.0
            optimizer.zero_grad()
            start_time = time.time()

            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                candidate_mask = batch["candidate_mask"].to(self.device)
                target_choice = batch["target_choice"].to(self.device)
                target_score = batch["target_score"].to(self.device)
                target_action = batch["target_action"].to(self.device)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16 if use_bf16 else torch.float16):
                    outputs = self.model(input_ids, attention_mask, candidate_mask)
                    
                    l_choice = self.criterion_choice(outputs["choice_logits"], target_choice)
                    l_score = self.criterion_score(outputs["score"], target_score)
                    l_action = self.criterion_action(outputs["action_logits"], target_action)

                    loss = (
                        self.cfg.loss_weight_choice * l_choice +
                        self.cfg.loss_weight_score * l_score +
                        self.cfg.loss_weight_action * l_action
                    )
                    loss = loss / self.cfg.gradient_accumulation_steps

                if use_bf16:
                    loss.backward()
                else:
                    scaler.scale(loss).backward()

                if (step + 1) % self.cfg.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                    if use_bf16:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                    else:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()

                    scheduler.step()
                    optimizer.zero_grad()

                running_loss += loss.item() * self.cfg.gradient_accumulation_steps

            avg_train_loss = running_loss / len(train_loader)
            epoch_time = time.time() - start_time

            # Doğrulama (Evaluation)
            val_metrics = self.evaluate(val_loader)
            console.print(
                f"[bold]Epoch {epoch}/{self.cfg.num_train_epochs}[/bold] ({epoch_time:.1f}s) - "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Choice Acc: [green]{val_metrics['choice_acc']:.2%}[/green] | "
                f"Score MAE: {val_metrics['score_mae']:.4f} | "
                f"Action F1: {val_metrics['action_f1']:.2%}"
            )

            if val_metrics['choice_acc'] > best_val_acc:
                best_val_acc = val_metrics['choice_acc']
                self.save_model(os.path.join(self.cfg.output_dir, "best"))

        # Eğitim sonrası Sıcaklık Kalibrasyonu (Temperature Calibration)
        console.print("\n[bold yellow]Laya Sıcaklık Kalibrasyonu (Temperature Scaling) Başlatılıyor...[/bold yellow]")
        self.calibrate_temperatures(val_loader)
        self.save_model(os.path.join(self.cfg.output_dir, "best_calibrated"))

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_pred_choices, all_true_choices = [], []
        all_pred_scores, all_true_scores = [], []
        all_pred_actions, all_true_actions = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                candidate_mask = batch["candidate_mask"].to(self.device)

                outputs = self.model(input_ids, attention_mask, candidate_mask)
                
                preds = torch.argmax(outputs["choice_logits"], dim=-1).cpu().numpy()
                all_pred_choices.extend(preds)
                all_true_choices.extend(batch["target_choice"].numpy())

                all_pred_scores.extend(outputs["score"].cpu().numpy())
                all_true_scores.extend(batch["target_score"].numpy())

                action_probs = torch.sigmoid(outputs["action_logits"]).cpu().numpy()
                all_pred_actions.extend((action_probs >= 0.5).astype(int))
                all_true_actions.extend(batch["target_action"].numpy().astype(int))

        return {
            "choice_acc": accuracy_score(all_true_choices, all_pred_choices),
            "score_mae": mean_absolute_error(all_true_scores, all_pred_scores),
            "action_f1": f1_score(all_true_actions, all_pred_actions, zero_division=0),
            "brier_score": brier_score_loss(all_true_actions, all_pred_actions)
        }

    def calibrate_temperatures(self, val_loader: DataLoader):
        """Doğrulama kümesi üzerinde aşırı güveni (overconfidence) önlemek için Platt Scaling uygular."""
        self.model.eval()
        choice_logits_list, choice_labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                candidate_mask = batch["candidate_mask"].to(self.device)
                outputs = self.model(input_ids, attention_mask, candidate_mask)
                choice_logits_list.append(outputs["choice_logits"].cpu())
                choice_labels_list.append(batch["target_choice"].cpu())

        all_logits = torch.cat(choice_logits_list, dim=0)
        all_labels = torch.cat(choice_labels_list, dim=0)

        # Sıcaklık parametresi optimizasyonu
        temperature = nn.Parameter(torch.ones(1) * 1.5)
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
        criterion = nn.CrossEntropyLoss()

        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(all_logits / temperature, all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        optimal_t = float(torch.clamp(temperature, 0.1, 5.0).item())
        self.model.temperature_choice.data.fill_(optimal_t)
        console.print(f"[bold green]✓ Optimize Edilmiş Karar Sıcaklığı (T_choice): {optimal_t:.4f}[/bold green]")

    def save_model(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        # Checkpoint sözlüğü
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "options_vocab": self.all_options,
            "base_model_name": self.cfg.base_model_name,
            "temperature_choice": self.model.temperature_choice.item(),
            "temperature_action": self.model.temperature_action.item()
        }, os.path.join(save_path, "laya_turkish_model.pt"))
        
        # Tokenizer'ı kaydet
        self.tokenizer.save_pretrained(save_path)
        console.print(f"[green]Model ve Tokenizer kaydedildi: {save_path}[/green]")

if __name__ == "__main__":
    t_cfg = TrainingConfig()
    g_cfg = GenerationConfig()
    trainer = LayaTrainer(t_cfg, g_cfg)
    
    # data/train.jsonl mevcutsa eğitimi başlat
    if os.path.exists("data/train.jsonl"):
        trainer.train("data/train.jsonl", "data/val.jsonl")
    else:
        console.print("[yellow]data/train.jsonl bulunamadı. Lütfen önce synthetic_generator.py çalıştırın![/yellow]")
