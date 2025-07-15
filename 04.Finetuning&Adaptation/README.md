### **IV. Finetuning & Adaptation (파인튜닝 및 적응)**

*사전 학습된 모델을 특정 목적에 맞게 '조련'하는 과정입니다. 현업에서 가장 많이 활용되는 기술들입니다.*

1. **Finetuning Data (파인튜닝 데이터)**
    - Instruction Datasets (`instruction`, `output` 쌍)
    - Preference Datasets (`chosen`, `rejected` 쌍)
    - Data Quality & Curation
2. **Full Finetuning (SFT / Instruction Tuning)**
3. **Parameter-Efficient Fine-Tuning (PEFT)**
    - LoRA, QLoRA, Adapter Tuning, Prompt/Prefix Tuning
4. **Aligning with Human Preferences**
    - RLHF (Reward Modeling & PPO)
    - DPO (Direct Preference Optimization)