---
tags: [nca-genl, nvidia/responsible-ai, nvidia/llm-genai, evaluation, llm]
aliases: [LLM Evaluation Metrics, Truthfulness Score, Perplexity, BLEU Score, TruthfulQA, LLM benchmarks]
---

# LLM Evaluation Metrics

> **Domain**: Responsible AI / LLM Foundations
> **Cert Relevance**: NCA-GENL — Responsible AI (~8%) + LLM Foundations (~25%)
> **Related**: [[RAG Architecture]], [[NVIDIA NIM]], [[Training vs Inference]], [[NeMo Guardrails]]

## Overview

LLM evaluation requires multiple specialised metrics because no single number captures all dimensions of output quality. Each metric measures one specific property: factual correctness, linguistic fluency, similarity to a reference, or throughput. The most common exam and real-world trap is using the wrong metric for the wrong goal — particularly confusing fluency (perplexity) with factual accuracy (truthfulness). For production deployments where being wrong has real consequences, the Truthfulness Score is the correct primary metric.

---

## The Four Core Metrics

### Truthfulness Score

**What it measures**: Whether the LLM's generated content is factually correct — aligned with verifiable real-world facts.

**What problem it solves**: **Hallucination** — the tendency of LLMs to generate plausible-sounding but factually incorrect content. An LLM can produce fluent, confident text that is entirely fabricated. Truthfulness is the only metric in this set that directly detects this.

**How it works**: The model's output is compared against a trusted external source — a verified knowledge base, fact-checking service, or human expert review. The score reflects how much of the generated content can be verified as true.

**Primary benchmark**: **TruthfulQA** — 817 questions designed to probe areas where LLMs commonly hallucinate (health myths, conspiracy theories, common misconceptions).

**When to use**:
- Medical Q&A systems
- Legal information assistants
- Financial advice tools
- Scientific summarisation
- Any application where factual errors have real consequences

> [!tip] Exam Tip
> When an exam question asks which metric measures **factual accuracy** or addresses **hallucination**, the answer is always **Truthfulness Score**. No other metric in this set evaluates whether content is actually true.

---

### Perplexity

**What it measures**: How well the model predicts a sequence of text — a proxy for linguistic fluency and naturalness.

**Interpretation**: Lower perplexity = model found the text more predictable = text is more fluent and natural-sounding for that model.

**What it does NOT measure**: Factual correctness. A model can generate perfectly fluent, grammatically flawless sentences that are completely false. Perplexity has no mechanism to detect this.

**Formula (conceptual)**:
```
Perplexity = 2^(average cross-entropy loss per token)
Lower = better = more fluent
```

**When to use**: Comparing model fluency, evaluating text quality for creative or conversational applications, language model benchmarking.

> [!warning]
> Perplexity is a measure of **language quality**, not **factual quality**. A model with low perplexity that hallucinates is a fluent liar. Do not use perplexity to evaluate factual accuracy — this is the most common exam distractor on this topic.

---

### BLEU Score (Bilingual Evaluation Understudy)

**What it measures**: N-gram overlap between generated text and one or more human-written reference texts — originally designed for machine translation.

**How it works**: Counts how many word sequences (n-grams) in the generated output also appear in the reference. Higher overlap = higher BLEU.

**When to use**:
- Machine translation quality
- Summarisation with a gold-standard reference
- Tasks where there is one correct or expected output form

**Critical limitations**:
- Only compares against the reference — if the reference is wrong, a high BLEU means you matched a wrong answer
- Penalises valid paraphrases — multiple correct ways to express an answer all score low if they don't match the reference wording
- Does not verify factual correctness independently

> [!warning]
> BLEU measures **similarity to a reference**, not **truthfulness**. A high BLEU score on a factually incorrect reference means the model successfully reproduced an error. For open-ended generation tasks with many valid answers, BLEU is inappropriate.

---

### Token Generation Speed

**What it measures**: Throughput — tokens produced per second.

**What it does NOT measure**: Anything about output quality, accuracy, or fluency. Purely a performance and infrastructure metric.

**When to use**: Infrastructure sizing, latency SLA planning, user experience benchmarking, cost optimisation.

> [!warning]
> Token generation speed has **zero relationship** to output quality. A fast model that hallucinates is worse than a slow model that is accurate. Never use speed as a quality metric.

---

## Comparison Table

| Metric | Measures | Does it detect hallucination? | Use case |
|---|---|---|---|
| **Truthfulness Score** | Factual correctness | ✅ Yes — directly | Safety-critical applications |
| **Perplexity** | Linguistic fluency | ❌ No | Language quality assessment |
| **BLEU Score** | Reference similarity | ❌ No | Translation, fixed-reference tasks |
| **Token generation speed** | Throughput | ❌ No | Infrastructure sizing |

---

## The Hallucination Problem

Hallucination occurs because LLMs predict statistically likely next tokens — they do not retrieve verified facts from a database. The model has no internal truth-checking mechanism. It generates what *sounds* right given its training distribution, not what *is* right.

This is why Truthfulness Score matters: it is the only metric that evaluates the gap between "sounds right" and "is right."

**Mitigation strategies** (beyond just measuring):
- **[[RAG Architecture]]** — ground responses in retrieved verified documents
- **[[NeMo Guardrails]]** — constrain model outputs at the application layer
- **Human-in-the-loop review** — expert validation for high-stakes outputs
- **Confidence calibration** — train the model to express uncertainty when it doesn't know

---

## Additional Evaluation Metrics (Beyond the Core Four)

| Metric | Measures |
|---|---|
| **ROUGE** | Recall-oriented n-gram overlap — used in summarisation |
| **BERTScore** | Semantic similarity using BERT embeddings — better than BLEU for paraphrase tolerance |
| **Human evaluation** | Holistic quality via human raters — gold standard but expensive |
| **Faithfulness** | In RAG: does the response stay faithful to retrieved context? |
| **Answer relevance** | In RAG: does the response actually answer the question? |
| **Context precision/recall** | In RAG: did retrieval fetch the right documents? |

> [!note]
> For RAG-specific evaluation, the RAG Triad — **faithfulness, answer relevance, context precision** — is the standard framework. Truthfulness Score applies to general LLM factual accuracy; the RAG Triad evaluates the retrieval + generation pipeline specifically.

---

> [!note] Cybersecurity Connection
> These four metrics map precisely to different security testing disciplines. **Truthfulness Score** = vulnerability scanning — checks whether the content is actually safe/correct, not just well-formed. **Perplexity** = linting or static analysis — confirms syntactic correctness but says nothing about logical soundness or security. **BLEU** = diff/comparison tooling — measures similarity between two artefacts without evaluating whether either is safe. **Token generation speed** = load testing / benchmarking — measures throughput with no connection to correctness. Running a linter on malicious code gives it a clean bill of health; running perplexity on a hallucinating model does the same. The lesson in both domains: choose the right tool for what you actually want to verify.

---

## Related Notes

- [[RAG Architecture]] – RAG is a primary mitigation for hallucination; has its own evaluation triad
- [[NeMo Guardrails]] – application-layer constraint system to limit harmful or factually dangerous outputs
- [[Training vs Inference]] – evaluation metrics apply at inference time; training metrics (loss, accuracy) are different
- [[NVIDIA NIM]] – production NIM deployments require evaluation frameworks to validate output quality before deployment

---

## Key Mental Model

Think of LLM evaluation as a **security audit with multiple scanners**. Each scanner checks one specific thing: truthfulness checks for lies, perplexity checks for grammar, BLEU checks for plagiarism against a reference, speed checks for performance. No single scanner catches everything. A model can pass all three non-truthfulness checks with flying colours and still be factually dangerous. For production deployments where accuracy matters, truthfulness is the scanner you can't skip.

> [!tip] Exam Tip
> The NCA-GENL exam will present scenarios asking which metric to use for a given goal. The answer map: **factual accuracy / hallucination detection → Truthfulness Score**. **Fluency / language quality → Perplexity**. **Similarity to reference / translation quality → BLEU**. **Speed / throughput → Token generation speed**. These are mutually exclusive — no metric does double duty.
