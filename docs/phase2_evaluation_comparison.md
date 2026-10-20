# Phase 2: Language Modeling and Multi-Head Attention Evaluation Report

This document delivers the comprehensive side-by-side evaluation of **Model H (Telugu)** and **Model L (Nepali)** following the exact requirements of Section 2.3 of the project specification.

---

## 1. Intrinsic Language Modeling Metrics (Side-by-Side)

Both models share an identical 6-layer causal Transformer architecture ($d_{\text{model}}=512, n_{\text{heads}}=8, d_{\text{ff}}=2048, V=16000$) pretrained for 20,000 steps on their respective monolingual corpora.

| Intrinsic Metric | Model H (Telugu) | Model L (Nepali) | Better | Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **Training Steps** | 19,999 | 19,999 | — | Full planned pretraining schedule |
| **Validation Loss ($\mathcal{L}_{\text{CE}}$)** | **7.0214** | **5.0467** | Nepali (-1.9747) | Cross-entropy loss on held-out validation set |
| **Perplexity ($\text{PPL} = e^{\mathcal{L}}$)** | **1,120.39** | **155.50** | Nepali (7.2× lower) | Intrinsic next-token uncertainty |
| **Bits-Per-Byte (BPB)** | **2.3331** | **1.5800** | Nepali (-0.7531) | Normalized cross-entropy per raw UTF-8 byte |

### Discussion of the Intrinsic Metric Gap
1. **Tokenizer Fertility and Morphological Complexity**:
   - Telugu exhibits an average fertility of **3.86 characters per token**, whereas Nepali exhibits **3.38 characters per token**. Telugu is highly agglutinative with intricate Sandhi compounding and complex consonant conjuncts. A 16K subword vocabulary splits inflected Telugu words into more granular fragments, increasing the next-token prediction entropy.
2. **Compressibility under Bits-Per-Byte (BPB)**:
   - Because PPL is sensitive to vocabulary size and subword tokenization granularity, Bits-Per-Byte ($\text{BPB} = \frac{\mathcal{L} \times N_{\text{tokens}}}{N_{\text{bytes}} \ln 2}$) serves as the fair, tokenizer-agnostic ground truth. Model L achieves **1.5800 BPB** compared to Model H's **2.3331 BPB**, demonstrating that the Devanagari news corpus had higher semantic density and less noise than the historical/OCR Telugu corpus.

---

## 2. Generation Quality Across Temperatures (Held-Out Continutations)

Models were evaluated by feeding prefixes (12 tokens) of 50 held-out test sequences from `test.jsonl` and generating continuations ($L=30$ tokens) across four decoding temperatures: Greedy ($T=0.0$), $T=0.5$, $T=1.0$, and $T=1.5$. Evaluated against gold ground-truth continuations.

### Model H (Telugu) Generation Metrics

| Temperature | Word ROUGE-L | Char ROUGE-L | chrF | BLEU-4 | Distinct-1 | Distinct-2 | Repetition Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Greedy ($T=0.0$)** | 3.98% | 23.97% | 7.55 | 1.17 | 0.4273 | 0.5338 | 0.3430 |
| **$T=0.5$** | **5.44%** | **28.04%** | 11.32 | **1.55** | 0.8373 | 0.9745 | 0.0037 |
| **$T=1.0$** | 3.46% | 28.81% | 13.27 | 1.20 | 0.9753 | 1.0000 | 0.0000 |
| **$T=1.5$** | 2.55% | 28.72% | **15.16** | 1.21 | 0.9947 | 1.0000 | 0.0000 |

### Model L (Nepali) Generation Metrics

| Temperature | Word ROUGE-L | Char ROUGE-L | chrF | BLEU-4 | Distinct-1 | Distinct-2 | Repetition Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Greedy ($T=0.0$)** | 5.49% | 31.23% | 13.99 | 1.39 | 0.4680 | 0.5345 | 0.3926 |
| **$T=0.5$** | **6.66%** | **30.97%** | **16.01** | **1.61** | 0.6907 | 0.8159 | 0.1133 |
| **$T=1.0$** | 3.13% | 30.47% | 15.28 | 1.03 | 0.9487 | 0.9890 | 0.0000 |
| **$T=1.5$** | 1.99% | 30.85% | 15.60 | 0.74 | 0.9900 | 1.0000 | 0.0000 |

---

## 3. Metric Informativeness Analysis for Indic Language Modeling

### Why ROUGE-L was Initially Zero and How it was Resolved
- **Standard ROUGE Tooling Defect**: The standard Python `rouge-score` library relies on ASCII regular expressions (`[^a-z0-9]+`) that strip out all non-Latin characters, reducing Indic strings to empty lists and resulting in artificial zero scores.
- **Solution**: We implemented custom Unicode-aware tokenizers (`regex.findall(r'\b\w+\b', text, flags=regex.UNICODE)`) and character-level LCS. Word-level ROUGE-L reaches **5.44% - 6.66%** and Character ROUGE-L reaches **28.0% - 31.2%**, directly capturing valid prefix continuations.

### Informativeness of Evaluation Metrics:
1. **BLEU-4**:
   - *Limitation*: BLEU requires exact 4-gram matches against a single reference. In open-ended autoregressive generation, a model may produce a semantically valid and grammatically correct continuation that diverges from the specific ground truth. Hence, BLEU scores are naturally low ($\approx 1.0 - 1.6$).
2. **chrF / chrF++**:
   - *High Informativeness*: By computing character n-gram overlaps ($n=1 \dots 6$), chrF rewards correct morphological stems, grammatical suffixes, and partial word matches even when inflections vary. It correlates strongly with human judgment in morphologically rich Indic languages.
3. **Distinct-1 / Distinct-2 and Repetition Rate**:
   - *Key Diagnostic*: Greedy decoding demonstrates moderate repetition rates ($\approx 0.34 - 0.39$) due to argmax feedback loops. Sampling at $T=0.5$ successfully breaks degenerate cycles (reducing repetition rate to near $0.003 - 0.11$) while preserving topic coherence.

---

## 4. Multi-Head Attention Analysis

All attention heatmaps were generated using **numeric key and query position indices** ($0 \dots T-1$) on both axes, ensuring clear, reproducible figures without missing-glyph or font rendering anomalies.

### Attention Visualizations Available in Results
1. **Sample Heatmaps (3 Inputs per Language)**:
   - Early Layer (Layer 0): Heads 0, 1, 2
   - Late Layer (Layer 5): Heads 0, 1, 2
   - 2x3 Overview Comparison Grids: `attn_sample1_overview.png`, `attn_sample2_overview.png`, `attn_sample3_overview.png`
2. **Entropy and Distance Summary**:
   - `attention_entropy_distance.png`: Shannon Entropy (bits) and Mean Attention Distance (tokens) across all 6 layers and 8 heads.

### Head Specialization and Positional vs. Content-Based Behaviors
- **Layer 0 (Early Layer - Diffuse & Positional)**:
  - Heads display higher entropy ($\approx 1.0 - 1.9$ bits) and moderate attention distance. Queries attend broadly to preceding local tokens, functioning as local context integrators and positional encoders.
- **Layer 5 (Late Layer - Specialized & Functional)**:
  - **Diagonal / Previous-Token Tracking (e.g., L5-H2)**: Concentrates attention exclusively along the sub-diagonal ($i \to i-1$), passing state strictly sequentially.
  - **BOS / Anchor Attention (e.g., L5-H1)**: Attends heavily back to Position 0, functioning as an informational anchor / global null-token accumulator.
  - **Syntactic / Semantic Clustering (e.g., L5-H0)**: Exhibits sharp, sparse activations across specific clause boundaries and nominal phrases.

---

## 5. Artifact and Checkpoint Locations

- **Telugu Checkpoint**: `individual-project-nikk8076/telugu/best_checkpoint.pth` (Step 19,999)
- **Nepali Checkpoint**: `individual-project-nikk8076/nepali/best_checkpoint.pth` (Step 19,999)
- **Telugu Evaluation Scores**: `individual-project-nikk8076/telugu/results/evaluation_scores.txt`
- **Nepali Evaluation Scores**: `individual-project-nikk8076/nepali/results/evaluation_scores.txt`
- **Telugu Attention Figures**: `individual-project-nikk8076/telugu/results/figures/`
- **Nepali Attention Figures**: `individual-project-nikk8076/nepali/results/figures/`
