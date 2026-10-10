# PROJECT STATE & PERSISTENCE RECORD

**Project Name:** Turkish Decision Model (TDM)  
**Repository:** https://github.com/hsnsahinhsn/turkish-decision-model.git  
**Target Environment:** Cloud GPU (NVIDIA RTX PRO 6000 96GB / RTX 5090 32GB / RTX 4090 24GB / H200)  
**Primary Goal:** Sub-15 ms latency, highly calibrated, single-pass non-autoregressive decision & triage engine for autonomous Turkish agent workflows.

---

## 1. System Architecture

- **Backbone Architecture:** Non-autoregressive Transformer encoder (`convaiinnovations/laya-multilingual` / mmBERT-base backbone, 322M parameters).
- **Multi-Task Typed Decision Head:** Single forward pass simultaneously computes:
  1. `choice`: Categorical decision selected across dynamically passed candidate actions (via cross-attention / dot-product with candidate representations and dynamic candidate masking).
  2. `score`: Calibrated confidence / urgency / risk metric bounded in `[0.00, 1.00]`.
  3. `action_required`: Boolean intervention flag for agentic human-in-the-loop escalation or immediate tool execution.
- **Calibration Engine:** Platt Scaling with learned temperature scaling ($T_{\text{choice}}, T_{\text{action}}$) optimized on validation set to minimize Expected Calibration Error (ECE) and Brier Score.
- **Latency Target:** $< 15\text{ ms}$ on 512-token context under TensorRT / FP16 / BF16 execution.

---

## 2. Universal Macro-Domain Taxonomy (32 Domains)

To maximize marginal utility and eliminate cold-start bias, synthetic generation targets **750 samples per category** (**24,000 samples total**):

1. `banking_fintech`: EFT, kart blokaj, kredi, döviz, şüpheli işlem.
2. `e_commerce_retail`: İade, kargo takibi, beden değişimi, ödeme hatası, kupon.
3. `telecom_isp`: Kota aşımı, taahhüt, fiber arıza, fatura itirazı, numara taşıma.
4. `healthcare_telemedicine`: Randevu iptali, reçete yenileme, lab sonucu, triyaj, sigorta onayı.
5. `insurance_insurtech`: Kasko hasar bildirimi, poliçe zeyilname, acil sağlık teminatı.
6. `legal_compliance`: KVKK talebi, cayma hakkı, sözleşme maddesi inceleme, tahkim.
7. `airline_travel_hospitality`: Uçuş değişikliği, bagaj kaybı, check-in hatası, otel no-show.
8. `utilities_energy`: Elektrik/doğalgaz arıza bildirimi, sayaç okuma, abonelik devir.
9. `automotive_fleet`: Periyodik bakım, yol yardımı, telematik arıza kodu, filo rotalama.
10. `education_edtech`: Burs başvurusu, sınav itirazı, LMS erişim, sertifika doğrulama.
11. `public_services_egov`: e-Devlet kimlik doğrulama, belge sorgulama, vergi takvimi, belediye.
12. `logistics_freight`: Konteyner takibi, gümrük takılması, rota sapması, soğuk zincir alarmı.
13. `saas_b2b`: API rate limit, SSO yetkilendirme, webhook arızası, lisans yükseltme.
14. `cybersecurity_soc`: Phishing bildirimi, 2FA sıfırlama, yetkisiz IP erişim engeli, SIEM alarmı.
15. `food_delivery_quickcommerce`: Eksik sipariş, kurye gecikmesi, bozuk gıda şikayeti, alerjen kontrolü.
16. `entertainment_gaming`: Hesap çalınması, mikro-ödeme hatası, moderasyon itirazı, sunucu ping.
17. `real_estate_proptech`: Kira sözleşmesi yenileme, depozito iadesi, tapu randevusu, ekspertiz.
18. `human_resources_hrms`: Yıllık izin talebi, bordro sorusu, istifa süreci, avans talebi.
19. `manufacturing_iot`: PLC sıcaklık uyarısı, kestirimci bakım sinyali, vardiya stok eksiği.
20. `agriculture_agritech`: Toprak nem uyarısı, zirai don riski, gübreleme takvimi, hibeler.
21. `cryptocurrency_web3`: Ağ fee optimizasyonu, swap kayması, akıllı sözleşme uyarısı, cüzdan yetkilendirme.
22. `pharmaceutical_biotech`: İlaç yan etki bildirimi, soğuk zincir protokolü, muadil kontrolü.
23. `consumer_electronics`: Garanti kapsamı, donanım arızası, firmware güncelleme kilitlenmesi.
24. `media_publishing`: Telif ihlali bildirimi, basın bülteni yayını, abonelik iptali.
25. `procurement_supply_chain`: RFQ onayı, tedarikçi teklifi, sevk irsaliyesi, vade müzakeresi.
26. `personal_productivity`: Toplantı çakışması, görev önceliklendirme, e-posta taslak triyajı.
27. `crisis_management`: Doğal afet acil bildirim, tahliye protokolü, kriz masası eskalasyonu.
28. `charity_ngo`: Bağış dekontu, gönüllü koordinasyonu, afet yardım çağrısı, şeffaflık raporu.
29. `sports_fitness`: Antrenman iptali, üyelik dondurma, sakatlık bildirimi, PT randevusu.
30. `fashion_luxury`: Özel dikim randevusu, orijinallik sertifikası, VIP stilist talebi.
31. `environmental_waste`: Sıfır atık beyanı, tehlikeli atık bertarafı, karbon ayak izi raporu.
32. `aviation_maintenance`: Uçak arıza kaydı, MEL (Minimum Equipment List) kontrolü, parça değişimi.

---

## 3. Repository File Map

| Dosya | Açıklama |
|---|---|
| `config.py` | 32 domain tanımları, hiperparametreler, mimari konfigürasyonu. |
| `model.py` | `TurkishDecisionModel`, `DecisionHead`, maskeleme ve Platt scaling mimarisi. |
| `synthetic_generator.py` | Async vLLM/OpenAI formatlı 24k sentetik veri üretim motoru. |
| `train.py` | PyTorch + bfloat16 + FlashAttention-2 eğitim ve kalibrasyon döngüsü. |
| `serve.py` | FastAPI tabanlı, sub-15ms yüksek throughput çıkarım servisi (`/decide`). |
| `agentic_router_example.py` | LangGraph / CrewAI tarzı otonom karar orkestrasyonu demo kodu. |
| `README.md` | Kurumsal, profesyonel teknik dokümantasyon ve benchmark tablosu. |
| `.gitignore` | Temiz repo disiplini (checkpoints, bat scriptleri ve cache engeli). |
| `PROJECT_STATE.md` | Kalıcı hafıza ve sistem durum kaydı (bu dosya). |

---

## 4. Execution Workflow (When GPU Server is Ready)

1. **Ortam Kurulumu:**
   ```bash
   git clone https://github.com/hsnsahinhsn/turkish-decision-model.git
   cd turkish-decision-model
   pip install -r requirements.txt
   ```

2. **vLLM Sentetik Veri Motorunu Başlatma:**
   ```bash
   python -m vllm.entrypoints.openai.api_server \
       --model Qwen/Qwen3.5-9B-Instruct \
       --tensor-parallel-size 1 \
       --port 8000 \
       --max-model-len 4096
   ```

3. **Sentetik Veri Üretimini Başlatma (24,000 Örnek):**
   ```bash
   python synthetic_generator.py
   ```

4. **Model Eğitimi ve Sıcaklık Kalibrasyonu:**
   ```bash
   python train.py
   ```

5. **Servise Alma ve Çıkarım:**
   ```bash
   python serve.py
   ```
