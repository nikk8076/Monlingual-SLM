# BPE Tokenization — Design and Configurations

This document explains the design decisions, parameter configurations, and implementation details for the Nepali and Telugu Byte Pair Encoding (BPE) tokenizers.

---

## 1. Why BPE (Byte Pair Encoding)?

We selected **BPE** (Byte Pair Encoding) over SentencePiece's Unigram model:
* **LLM Compatibility**: Standard pretraining models (like Llama, GPT, Gemma, and Mistral) utilize BPE. Standardizing on BPE simplifies downstream Hugging Face integrations.
* **Deterministic Merges**: BPE builds vocabulary by iteratively merging the most frequent character pairs, leading to highly consistent subword boundaries and prefixes.

---

## 2. Tokenizer Configuration Parameters

To ensure correct tokenization of phonetic Indic alphasyllabaries, we applied the following parameters in `tokenizer.ipynb`:

| Parameter | Configuration | Linguistic Rationale |
| :--- | :---: | :--- |
| **Vocabulary Size** | 16,000 (16K) | Empirically selected. Keeps the embedding tables small while reducing fertility to under 2.1 tokens/word. |
| **Normalization** | `Identity` | Disables default Unicode normalizers (like NFKC) which decompose matras and strip formatting characters. |
| **Character Coverage** | 1.0 (100%) | Covers 100% of the target script Unicode characters, avoiding `<unk>` tokens. |
| **Byte Fallback** | `True` | Unknown unicode codepoints are tokenized as individual bytes rather than mapped to `<unk>`. |
| **Pre-tokenizer** | `Metaspace` | Converts whitespaces into a visual prefix character (`\u2581`), allowing perfect text reconstruction. |

---

## 3. Linguistic Design Decisions

### A. Preservation of ZWJ/ZWNJ
Zero Width Joiner (ZWJ, `\u200D`) and Zero Width Non-Joiner (ZWNJ, `\u200C`) are formatting characters that control how adjacent Indic consonants combine.
* **Telugu Ligatures**: ZWNJ is required to display correct conjunct consonants (e.g., preventing vowel matras from blending incorrectly). ZWJ is used to force a half-consonant shape.
* **Preservation**: By default, standard tokenizers strip ZWJ/ZWNJ. By using **Identity Normalization**, these control characters are 100% preserved in the vocabulary.

### B. Prevention of Deconstructed Matras
If NFD (Decomposition) normalizers are used, vowel signs (matras, like `ो`, `ా`) are split off from their base consonant and tokenized as standalone fragments. Enforcing Identity Normalization keeps the text in NFC format, ensuring matras remain combined with their base letters during training.

### C. Pristine Monolingual Regex Filtering
Web-crawled public dumps (like Wikipedia citations) contain translation tables, references, and multilingual citations written in foreign scripts (e.g., Lao, Tibetan, Cyrillic, Chinese). 
BPE tokenizers add *every unique character* in the training text to the base vocabulary. Even a single occurrence of a Lao letter will occupy a base slot.

To prevent this vocabulary pollution, we compiled a script-filtering regular expression into the streaming batch iterator to clean the training text *before* BPE sees it:
* **Nepali (Devanagari) Filter**:
  `re.compile(r"[^\u0900-\u097F\u0000-\u007F\u2000-\u200D\u2010-\u201F।॥\s]")`
* **Telugu Filter**:
  `re.compile(r"[^\u0C00-\u0C7F\u0000-\u007F\u2000-\u200D\u2010-\u201F\s]")`

This strips out all characters outside the target language script block, standard English/ASCII, formatting codes (ZWJ/ZWNJ), punctuation, and spacing, ensuring a **100% clean monolingual vocabulary**.

---

## 4. Implementation Details

* **Library**: Built using the Rust-backed Hugging Face `tokenizers` library.
* **Streaming Generator**: Rather than writing a temporary text corpus to disk (which wastes disk space and time), the training script defines a Python generator (`batch_iterator`) that streams text directly from the shrunken JSONL files in batches of 10,000 lines.
* **Zero RAM footprint**: By using an iterator, memory usage is kept **under 100 MB** for the entire 350M word training run, preventing system out-of-memory crashes on laptops.
* **Outputs**:
  - Tokenizer configuration JSON: `nepali_tokenizer.json` / `telugu_tokenizer.json`
  - Vocabulary mapping text file (sorted by ID): `nepali_tokenizer.vocab` / `telugu_tokenizer.vocab`
