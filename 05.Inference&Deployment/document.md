### **V. Inference & Deployment (추론 및 배포)**

*학습된 모델을 실제 서비스에서 효율적으로 사용하기 위한 기술들입니다.*

1. **Decoding Strategies (디코딩 전략)**
    - Greedy Search, Beam Search, Top-k/Top-p (Nucleus) Sampling
2. **Inference Optimization (추론 최적화)**
    - Quantization (양자화), Pruning (가지치기), Knowledge Distillation (지식 증류)
    - Speculative Decoding
3. **Serving & Infrastructure (서빙 및 인프라)**
    - Batching, PagedAttention, vLLM, TensorRT-LLM