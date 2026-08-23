# Telugu Dataset Collection — Phase 1 Summary

This folder contains the codebase and raw dataset pipeline for **Model H (Telugu)**.
All manual sources use raw Wikimedia XML dumps or Internet Archive OCR — **no Hugging Face for manual data**.

---

## 1. Non-Manual Sources (`datasets/raw/downloaded/`)

These are downloaded programmatically from public HuggingFace datasets and web crawls.

| File | Source | Content | Est. Words |
|:---|:---|:---|---:|
| `telugu_novels.jsonl` | `eswardivi/telugu_dataset` (HF) | Literary stories, novels, translated historical texts | ~8M |
| `telugu_ocr.jsonl` | `harsha-desaraju/telugu-ocr-text` (HF) | Transcribed prose from scan files | ~5M |
| `telugu_culturax.jsonl` | `LightFury9/CulturaX_500ksamples_8splits_te` (HF) | Crawled web pages, articles, blog posts | ~120M |
| `telugu_indiccorp.jsonl` | `ai4bharat/IndicCorpV2` (HF) | Large-scale web text corpus by AI4Bharat | ~180M |

### ⚠️ Known Data Issues (Non-Manual)
- **`telugu_novels.jsonl`**: Page markers (`--- Page N ---`), chapter dividers, occasional transliterated English phrases, inconsistent spacing around punctuation.
- **`telugu_ocr.jsonl`**: High frequency of OCR misread characters — numbers substituted for similar-looking Telugu glyphs, raw symbols (`@`, `#`, `|`), punctuation noise, garbled ligatures. Short stub pages (<50 chars) with no content.
- **`telugu_culturax.jsonl`**: Boilerplate web text — navigation bar headers, cookie notices, ad text, repeated taglines. Mixed English/Telugu paragraphs (code snippets, URLs, social media handles). Duplicate paragraphs across pages from the same domain.
- **`telugu_indiccorp.jsonl`**: Machine-translated content mixed with native Telugu. Sentence-level English code-switching. Inconsistent sentence boundary markers.

---

## 2. Manual Sources (`datasets/raw/manual/`)

These are collected exclusively via local XML dump parsing, Internet Archive OCR text files, and direct scraping — no HuggingFace datasets used.

| File | Source | Content | Est. Words |
|:---|:---|:---|---:|
| `telugu_wikipedia.jsonl` | `dumps.wikimedia.org/tewiki/latest/` | Encyclopedic articles — science, history, geography, culture | ~35M |
| `telugu_wikisource_manual.jsonl` | `dumps.wikimedia.org/tewikisource/latest/` | Proofread classical books, poetry, songs, historical documents | ~26M |
| `telugu_wikibooks.jsonl` | `dumps.wikimedia.org/tewikibooks/latest/` | Educational textbooks and reference material | ~3.3M |
| `telugu_wikiquote.jsonl` | `dumps.wikimedia.org/tewikiquote/latest/` | Quotations from literature and public figures | ~220K |
| `telugu_wiktionary.jsonl` | `dumps.wikimedia.org/tewiktionary/latest/` | Dictionary definitions, usage examples | ~4.5M |
| `telugu_vipula_magazine.jsonl` | Internet Archive `vipula-eenadu-group` | Vipula monthly magazine, 306 issues (1978–present) | ~22M |
| `telugu_newspapers.jsonl` | Internet Archive (Andhra Jyothi, Eenadu, Sakshi) | Contemporary Telugu daily newspaper editions | ~12M |
| `telugu_vignana_sarvasvam.jsonl` | Internet Archive `VignanaSarvasvamVolVVishwaSahithi` | Telugu encyclopedic reference (Vignana Sarvasvam Vol V) | ~1M |
| `telugu_puranas_classical.jsonl` | Internet Archive (Brahmanda, Visnu Puranam) | Classical puranic texts, estd. 1868 | ~170K |
| `bharatiya_chitrakala_ocr.jsonl` | Local Tesseract OCR on scanned PDF | Telugu art history book | ~22K |
| `navarasa_kadambari_ocr.jsonl` | Local Tesseract OCR on scanned PDF | Classical Telugu story anthology | ~2.6K |

### ⚠️ Known Data Issues (Manual)
- **`telugu_wikipedia.jsonl`**: Residual MediaWiki markup — `{{template}}`, `[[wikilinks]]`, `==Section Headers==`. Info-box tables with pipe-separated cell data. Reference citation markers like `[1]`, `[2]`. Some stubs with only 1–2 sentences.
- **`telugu_wikisource_manual.jsonl`**: Transclusion headers, navigation footers (`← prev | next →`), page break markers (`<pb/>`), chapter index tables. Some pages are pure Telugu numerals with no prose.
- **`telugu_wikibooks.jsonl`**: Template boilerplate at page start/end. Navigation boxes (`{{BookCat}}`). Heavily structured content with bullet lists that may not parse as natural prose.
- **`telugu_wiktionary.jsonl`**: Highly structured definition format. Each entry may have very short text (<20 words). Mixed with Sanskrit/Hindi loanword definitions in Devanagari script.
- **`telugu_vipula_magazine.jsonl`**: OCR-generated text from scanned magazine pages. Noise from scan quality variation across decades (1978 scans noisier than 2000s). Column-layout scan artifacts (words split mid-line). Occasional Devanagari characters from advertisements.
- **`telugu_newspapers.jsonl`**: OCR from newspaper scans — column bleed-through, ad text fragments, classified listings (very short, non-prose). Telugu and English headlines mixed. Date/edition headers on every page.
- **`telugu_puranas_classical.jsonl`**: 19th-century Telugu script variants that may differ from modern Unicode. Sandhi-split words, archaic grammar, Sanskrit verse interspersed with Telugu prose.
- **`bharatiya_chitrakala_ocr.jsonl` / `navarasa_kadambari_ocr.jsonl`**: Local Tesseract OCR quality depends on PDF scan resolution. Expect ~10–20% character error rate on lower-quality pages.

---

## 3. File Format

All datasets use **JSON Lines (.jsonl)** — one JSON object per line:
```json
{"text": "...", "source": "telugu_wikipedia_dump", "title": "..."}
```

---

## 4. Cleaning & Filtering Goals (Phase 2 — `clean.ipynb`)

| Problem | Planned Fix |
|:---|:---|
| MediaWiki markup (`{{}}`, `[[]]`, `<...>`) | Regex-based strip |
| OCR noise (garbled glyphs, symbols, numbers for chars) | Min Devanagari/Telugu char ratio filter per sentence |
| Non-Telugu sentences (English, Hindi code-switching) | Sentence-level script filter — drop lines with >30% ASCII alpha |
| Duplicate paragraphs (web crawl boilerplate) | MinHash or exact-substring deduplication |
| Very short stubs (<50 chars or <5 words) | Length threshold filter |
| Newspaper column bleed / classified ad fragments | Heuristic short-line density filter |
| Digit & script preservation | Keep ASCII `0-9` and Telugu `౦-౯`; normalize punctuation |
| Archaic 19th-century script variants | Unicode normalization (NFC) |
