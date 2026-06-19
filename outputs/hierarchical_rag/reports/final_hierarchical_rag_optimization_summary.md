# Hierarchical RAG Optimization Final Summary

## 1. Project Goal

This project builds an evaluation-driven Hierarchical RAG system for multi-hop question answering.

The system is designed to evaluate and optimize the full RAG pipeline, including retrieval, context packing, generation, groundedness, and final answer correctness.

The main task is HotpotQA-style factual question answering.

---

## 2. Retrieval-Side Optimization

The project first built a hierarchical retrieval pipeline:

- Golden evaluation set construction
- Parent document construction
- Child chunk construction
- Parent-child mapping
- BM25 child retrieval baseline
- Parent BM25 with child expansion
- BGE embedding retrieval
- Hybrid RRF grid search
- BGE reranking
- Final RAG context pack construction

The best retrieval-side result reached high context recall:

- Top10 Context Recall: 0.9467
- Top10 Context Hit: 1.0000

This showed that the retrieval system was already able to find answer-related evidence for most questions.

However, Top10 context precision remained low:

- Top10 Context Precision: 0.2100

This means the system retrieved enough evidence, but also included noisy contexts.

---

## 3. Context Pack Optimization

The project then optimized context packing.

Several variants were compared:

- top10_original_recheck
- top10_soft_cap2_compressed
- top7_soft_cap2_compressed

The goal was to reduce context noise while preserving answerability.

The final retained context packs were:

- `rag_inputs_v3_top10_original_recheck.jsonl`
- `rag_inputs_v3_top10_soft_cap2_compressed.jsonl`
- `rag_inputs_v3_top7_soft_cap2_compressed.jsonl`

The top10 variants preserved high recall, while the top7 compressed variant improved precision with some recall trade-off.

---

## 4. Qwen2.5-0.5B Generator Results

The initial local generator was Qwen2.5-0.5B-Instruct.

Even with high context recall, the 0.5B generator produced weak answer quality.

The best 0.5B v4 result on `top10_soft_cap2_compressed` was:

- Exact Match: 0.1600
- Contains Match: 0.2000
- Groundedness Proxy: 0.4400
- Soft Triad Pass: 0.2000

Diagnostics showed that the failure was not mainly caused by truncation or missing evidence. The model could often see the answer in the context, but failed to reliably extract or generate the correct short answer.

This indicated that the 0.5B generator was the main bottleneck.

---

## 5. Qwen2.5-7B No-Finetuning Baseline

The project then moved to an AutoDL RTX 4090D server and evaluated Qwen2.5-7B-Instruct without fine-tuning.

The strongest 7B baseline result was obtained by `top10_original_recheck`:

- Exact Match: 0.6400
- Contains Match: 0.7400
- Groundedness Proxy: 0.9600
- Soft Triad Pass: 0.7400

For `top10_soft_cap2_compressed`, the improvement over 0.5B was:

- Exact Match: 0.1600 -> 0.6400
- Contains Match: 0.2000 -> 0.7200
- Groundedness Proxy: 0.4400 -> 0.9400
- Soft Triad Pass: 0.2000 -> 0.7200

This confirmed that the main bottleneck had shifted from retrieval to generation capability.

The same retrieved contexts worked much better when used by a stronger generator.

---

## 6. RAG-SFT Data Construction

To further improve generator behavior, the project constructed RAG-SFT training data from HotpotQA train.

The generated supervised data contains:

- Train examples: 2000
- Dev examples: 200
- Contexts per example: up to 10
- Output target: gold short answer

The training objective was:

Input:

- question
- retrieved-like contexts

Output:

- concise gold answer

The goal was to improve the model's behavior in RAG settings, especially:

- concise answer generation
- context-grounded answer extraction
- reduced ungrounded generation
- reduced answer format errors
- better handling of short factual answers

---

## 7. Qwen2.5-7B QLoRA SFT

The project trained Qwen2.5-7B-Instruct with 4bit QLoRA.

Training settings:

- Base model: Qwen2.5-7B-Instruct
- Quantization: 4bit NF4
- LoRA r: 16
- LoRA alpha: 32
- Training examples: 2000
- Max steps: 500
- Epoch: 1.0

Training completed successfully:

- Train runtime: 1306 seconds
- Train loss: 0.1570
- Output adapter: `outputs/checkpoints/qwen25_7b_rag_sft_lora_full`

---

## 8. Baseline 7B vs LoRA SFT

After LoRA SFT, the model was evaluated on the same 50-example RAG evaluation set.

### top10_original_recheck

- Exact Match: 0.6400 -> 0.6400
- Contains Match: 0.7400 -> 0.8000
- Groundedness Proxy: 0.9600 -> 0.9800
- Soft Triad Pass: 0.7400 -> 0.8000

### top10_soft_cap2_compressed

- Exact Match: 0.6400 -> 0.6400
- Contains Match: 0.7200 -> 0.8000
- Groundedness Proxy: 0.9400 -> 0.9600
- Soft Triad Pass: 0.7200 -> 0.7800

### top7_soft_cap2_compressed

- Exact Match: 0.6200 -> 0.6600
- Contains Match: 0.7200 -> 0.8200
- Groundedness Proxy: 0.9400 -> 0.9600
- Soft Triad Pass: 0.7200 -> 0.8000

The best post-SFT result was:

- Variant: top7_soft_cap2_compressed
- Exact Match: 0.6600
- Contains Match: 0.8200
- Groundedness Proxy: 0.9600
- Soft Triad Pass: 0.8000

---

## 9. Final Conclusion

This project shows that a strong hierarchical RAG pipeline requires both retrieval-side optimization and generator-side optimization.

The retrieval system achieved high context recall, proving that relevant evidence could usually be retrieved.

However, the weak 0.5B generator could not reliably use the retrieved evidence.

Replacing the generator with Qwen2.5-7B caused a large improvement, confirming that generator capability was the major bottleneck after retrieval optimization.

Finally, RAG-SFT with QLoRA further improved answer coverage, groundedness, and soft triad pass.

The main improvement from SFT was not a dramatic Exact Match increase, but a more stable and grounded RAG answer style.

Overall, the best system configuration after all stages was:

- Retriever: optimized hierarchical retrieval with BGE/RRF/rerank context pipeline
- Context pack: top7_soft_cap2_compressed or top10_original_recheck depending on recall/precision trade-off
- Generator: Qwen2.5-7B-Instruct with RAG-SFT LoRA adapter
- Best post-SFT result: EM=0.6600, Contains=0.8200, Groundedness=0.9600, SoftTriad=0.8000

---

## 10. Next Possible Improvements

Future work can focus on:

- increasing RAG-SFT training data size
- adding hard negative contexts
- training with retrieved contexts instead of original HotpotQA distractor contexts
- adding preference optimization such as DPO
- using a stronger LLM-as-a-judge
- adding citation-level faithfulness evaluation
- evaluating on a larger HotpotQA validation subset
