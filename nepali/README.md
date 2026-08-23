# Nepali Dataset Collection — Phase 1 Summary

This folder contains the codebase and raw dataset pipeline for **Model L (Nepali)**.
All manual sources use raw Wikimedia XML dumps, Internet Archive OCR text files, or GitHub mirrors — **no Hugging Face for manual data**.

---

## 1. Non-Manual Sources (`datasets/raw/downloaded/`)

Downloaded programmatically from public HuggingFace datasets.

| File | Source | Content | Est. Words |
|:---|:---|:---|---:|
| `nepali_sangraha.jsonl` | `nepali-sangraha/sangraha` (HF) | Verified high-quality Nepali web text segments | ~120M |
| `nepali_sangraha_unverified.jsonl` | `nepali-sangraha/sangraha_unverified` (HF) | Unverified raw Nepali web crawl text | ~120M |
| `nepali_indiccorp.jsonl` | `ai4bharat/IndicCorpV2` `npi_Deva` split (HF) | AI4Bharat large-scale Nepali web corpus | ~100M |

### ⚠️ Known Data Issues (Non-Manual)
- **`nepali_sangraha.jsonl`**: While "verified", still contains some transliterated Romanized Nepali, stray HTML entities (`&amp;`, `&nbsp;`), and sentence fragments from incomplete crawl extraction. Some documents are very short (<3 sentences).
- **`nepali_sangraha_unverified.jsonl`**: High English code-switching (news portals with bilingual content). Boilerplate web fragments — cookie banners, navigation menus, social sharing text. Duplicate content across entries from the same source URL. Inconsistent punctuation (Western vs. Devanagari danda `।`).
- **`nepali_indiccorp.jsonl`**: Machine-translated content mixed with native Nepali. Mixed Hindi and Nepali (scripts identical — Devanagari — so hard to distinguish without langdetect). Sentence-level English code-switching common in tech/news domains.

---

## 2. Manual Sources (`datasets/raw/manual/`)

Collected via XML dump parsing, Internet Archive OCR, GitHub scrape archives, and local OCR — no HuggingFace datasets.

| File | Source | Content | Est. Words |
|:---|:---|:---|---:|
| `nepali_wikipedia_manual.jsonl` | `dumps.wikimedia.org/newiki/latest/` | Encyclopedic articles — history, geography, culture | ~9.4M |
| `nepali_wikibooks.jsonl` | `dumps.wikimedia.org/newikibooks/latest/` | Educational textbooks and learning material | ~502K |
| `nepali_wiktionary.jsonl` | `dumps.wikimedia.org/newiktionary/latest/` | Dictionary definitions and usage examples | ~1.5M |
| `nepali_brihat_sabdakosh.jsonl` | Internet Archive JSON dump (123K entries) | Pragya Nepali Brihat Sabdakosh — unabridged dictionary | ~1.4M |
| `nepali_classical_books.jsonl` | Internet Archive OCR (Vedas, Gitas, Law docs) | Bhagwat Mahapuran, Vedas, classical Devanagari texts | ~5.8M |
| `nepali_magazine_corpus.jsonl` | Internet Archive OCR (Nepal, Nari, Fagun magazines) | Weekly/monthly editorial magazines | ~4.2M |
| `nepali_news_manual.jsonl` | GitHub ZIP mirror (ashmitbhattarai/LSTM repo) | Nagarik & Setopati news articles (pre-2017) | ~2.9M |
| `nepali_16news.jsonl` | GitHub RAR (sndsabin/Nepali-News-Classifier) | 16-category news corpus from 10 Nepali portals | ~9M |
| `nepali_all_vedas.jsonl` | Internet Archive `all-ved-nepali` | All 4 Vedas in Nepali — Rig, Yajur, Sama, Atharva | ~3M |
| `nepali_purnima_magazine.jsonl` | Internet Archive `purnima-04-04_202407` | Purnima — 99-issue academic history quarterly | ~4.5M |
| `nepali_bhagwat_mahapuran2.jsonl` | Internet Archive `bhagwat-mahapuran-in-nepali_202203` | Bhagwat Mahapuran — full 7-volume edition | ~4M |
| `nepali_skanda_purana.jsonl` | Internet Archive `20230920_20230920_0856` | Himavatkhanda — Skanda Purana (Sanskrit + Nepali) | ~1.8M |
| `nepali_gorkhapatra.jsonl` | Internet Archive `gorkhapatra-2019-10-13` | Gorkhapatra national newspaper (17 issues, Oct 2019) | ~1.2M |
| `nepali_misc_books.jsonl` | Internet Archive (Satyarth Prakash, Nirnaya Sindhu, Swasthani, Medical, Khushi) | Diverse religious, legal, and literary texts | ~2M |
| `sumnima_ocr.jsonl` | Local Tesseract OCR (`lang=nep`) on scanned PDF | *Sumnima* — BP Koirala novel | ~500 |

### ⚠️ Known Data Issues (Manual)
- **`nepali_wikipedia_manual.jsonl`**: Residual MediaWiki markup — `{{template}}`, `[[wikilinks]]`, `==headers==`. Info-box pipe tables. Citation markers `[1]`, `[2]`. Stub articles with very short content.
- **`nepali_wikibooks.jsonl` / `nepali_wiktionary.jsonl`**: Template boilerplate. Wiktionary entries are heavily structured — definition lists, part-of-speech headers — not natural prose. Very short entries (<10 words each).
- **`nepali_brihat_sabdakosh.jsonl`**: Dictionary format — each entry is a short definition. Not natural running prose. Sanskrit loanwords and archaic vocabulary that may confuse the tokenizer if treated as body text.
- **`nepali_classical_books.jsonl`**: OCR-generated text from scanned PDFs and djvu files. Noise varies by source — older scans have higher glyph error rates. Sandhi-joined Sanskrit compounds intermixed with Nepali prose. Page headers and footers captured as text fragments.
- **`nepali_magazine_corpus.jsonl`**: Multi-column magazine layout artifacts in OCR — words split at column boundaries, hyphens at line ends. Advertisements captured as text blocks. Older editions (1970s–80s) have higher OCR error rates.
- **`nepali_news_manual.jsonl` / `nepali_16news.jsonl`**: Pre-2017 news text — some older encoding issues. Category/tag lines captured at start/end of article files. Occasional Romanized Nepali phrases in tech/sports articles.
- **`nepali_all_vedas.jsonl`**: Sanskrit Vedic verses interleaved with Nepali commentary — hard to separate. Verse numbering (e.g., `१.२.३`) captured as text. Very archaic Devanagari compounds that differ from modern Nepali.
- **`nepali_purnima_magazine.jsonl`**: Academic quarterly — dense Sanskrit/Nepali mixed prose. Footnotes and citation numbers embedded in text. Some issues have poor scan quality leading to OCR errors on Devanagari conjuncts.
- **`nepali_bhagwat_mahapuran2.jsonl` / `nepali_skanda_purana.jsonl`**: Heavily Sanskrit-influenced. Shloka (verse) interspersed with prose commentary. Page/chapter header fragments. Potential overlap with `nepali_classical_books.jsonl` for common puranas.
- **`nepali_gorkhapatra.jsonl`**: Newspaper OCR — column bleed, classified ad fragments, date/edition headers on every page. Mixed English headlines in international news sections.
- **`nepali_misc_books.jsonl`**: Diverse quality — Swasthani OCR is cleaner, medical texts have technical terminology and English drug names, Satyarth Prakash is polemical/philosophical prose.
- **`sumnima_ocr.jsonl`**: Local Tesseract OCR — very small file. Quality limited by PDF scan resolution.
- **Hindi/Nepali separation**: Since both use Devanagari script, Hindi text that slipped in cannot be detected by script alone — needs `langdetect` or character n-gram based filtering in Phase 2.

---

## 3. File Format

All datasets use **JSON Lines (.jsonl)** — one JSON object per line:
```json
{"text": "...", "source": "nepali_wikipedia_dump"}
```

---

## 4. Cleaning & Filtering Goals (Phase 2 — `clean.ipynb`)

| Problem | Planned Fix |
|:---|:---|
| MediaWiki markup (`{{}}`, `[[]]`, `==`, `<...>`) | Regex-based strip |
| OCR noise (garbled Devanagari glyphs, symbols) | Min Devanagari char ratio filter per sentence |
| Hindi/Nepali confusion (same script) | `langdetect` or fastText language ID filter |
| Non-Nepali sentences (English, Romanized) | Drop lines with >25% ASCII alpha characters |
| Sanskrit verse (shloka) with no Nepali prose | Heuristic: filter lines with >60% Sanskrit sandhi patterns |
| Duplicate paragraphs (web crawl boilerplate) | MinHash or exact-substring deduplication |
| Very short stubs (<50 chars or <5 words) | Length threshold filter |
| Newspaper column bleed / ad fragments | Short-line density + non-prose token ratio filter |
| Dictionary entries (very short, non-prose) | Optionally separate Sabdakosh into its own fine-tuning set |
| Encoding issues / broken Unicode | Unicode normalization (NFC) + broken-char ratio filter |
| Danda standardization | Normalize `।` and `॥` consistently |
