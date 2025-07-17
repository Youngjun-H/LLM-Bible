### **I. Foundations (기초 이론)**

*LLM의 근간을 이루는 핵심 아이디어와 기술들입니다. 모든 것은 여기서 시작됩니다.*

1. **Basic NLP Concepts (자연어 처리 기초)**
    - Tokenization (토큰화): BPE, WordPiece, SentencePiece
    - Embeddings (임베딩): Word2Vec, GloVe, FastText
2. **Pre-Transformer Era (트랜스포머 이전 시대)**
    - RNN (Recurrent Neural Network), LSTM & GRU
    - Vanishing/Exploding Gradients (기울기 소실/폭주 문제)
3. **The Game Changer (게임 체인저)**
    - Attention Mechanism (어텐션 메커니즘) & Seq2Seq Models

---

### **II. Core Architectures (핵심 아키텍처)**

*현대 LLM을 구성하는 뼈대와 주요 모델 계보를 이해합니다.*

1. **Transformer Architecture Deep Dive (트랜스포머 아키텍처 심층 탐구)**
    - Self-Attention & Multi-Head Attention
    - Positional Encoding (위치 인코딩)
    - Encoder-Decoder Structure (인코더-디코더 구조)
    - Layer Normalization & Residual Connections
2. **Major Model Families (주요 모델 계보)**
    - **Encoder-Only:** BERT, RoBERTa
    - **Decoder-Only:** GPT Series, LLaMA, Mistral
    - **Encoder-Decoder:** T5, BART

---

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

---

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

---

### **V. Inference & Deployment (추론 및 배포)**

*학습된 모델을 실제 서비스에서 효율적으로 사용하기 위한 기술들입니다.*

1. **Decoding Strategies (디코딩 전략)**
    - Greedy Search, Beam Search, Top-k/Top-p (Nucleus) Sampling
2. **Inference Optimization (추론 최적화)**
    - Quantization (양자화), Pruning (가지치기), Knowledge Distillation (지식 증류)
    - Speculative Decoding
3. **Serving & Infrastructure (서빙 및 인프라)**
    - Batching, PagedAttention, vLLM, TensorRT-LLM

---

### **VI. Evaluation (모델 평가)**

*모델의 성능을 어떻게 측정하고 비교할 것인가에 대한 문제입니다.*

1. **Standard Benchmarks (표준 벤치마크)**
    - GLUE, SuperGLUE, MMLU, HellaSwag, HumanEval
2. **Evaluation Metrics (평가 지표)**
    - Perplexity (PPL), BLEU, ROUGE
3. **Modern Evaluation Paradigms (최신 평가 패러다임)**
    - LLM-as-a-Judge, Human Evaluation

---

### **VII. Advanced & Frontier Topics (심화 및 최신 연구 분야)**

*LLM 연구의 최전선에 있는 뜨거운 주제들입니다.*

1. **Architectural Innovations:** Mixture of Experts (MoE)
2. **Expanding Context:** Long Context, Retrieval-Augmented Generation (RAG)
3. **Multimodality:** Vision-Language Models (VLM)
4. **LLM as Agents:** Tool Use, Planning & Reasoning (ReAct)
5. **Core Phenomena:** In-Context Learning (ICL), Emergent Abilities

---

### **VIII. Ethics & Safety (윤리 및 안전)**

*기술의 힘이 커질수록 책임도 커집니다. 모든 전문가가 깊이 고민해야 할 영역입니다.*

1. **Core Challenges:** Hallucination, Bias, Toxicity
2. **Mitigation & Alignment:** Safety Tuning, Red Teaming, Constitutional AI
3. **Societal Impact:** Copyright, Data Privacy, Explainability (XAI)
