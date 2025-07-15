### **III. Model Training (모델 학습)**

*거대 모델이 어떻게 '탄생'하는지를 다룹니다. 데이터부터 학습 기법까지의 전 과정입니다.*

1. **Pre-training Data (사전 학습 데이터)**
    - Data Collection, Curation, Preprocessing, Cleaning
2. **Pre-training (사전 학습)**
    - Training Objectives: MLM, CLM
    - Loss Functions & Optimizers
3. **Scaling Laws (스케일링 법칙)**
    - Model Size, Dataset Size, Compute 간의 관계
4. **Distributed Training (분산 학습)**
    - Data/Tensor/Pipeline Parallelism
    - Frameworks: DeepSpeed, Megatron-LM