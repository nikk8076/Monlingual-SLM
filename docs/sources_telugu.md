# Telugu Monolingual Dataset — List of Sources

This document lists all manual and public data sources aggregated to construct the 400M-word cleaned Telugu monolingual corpus.

---

## 1. Manual Sources (97.30 Million Words Post-Cleaning)

These sources represent high-quality regional data collected manually via web scraping, OCR extraction, and wiki parsing.

### A. Scraped News Portals
* **[Andhra Jyothy](https://www.andhrajyothy.com) (`telugu_news_andhrajyothy.jsonl`)**: Crawled articles from Andhra Jyothy news portal.
* **Newspapers Corpus (`telugu_newspapers.jsonl`)**: Aggregated text from various regional news outlets and newspapers.

### B. Periodicals \& Magazines
* **Vipula Magazine (`telugu_vipula_magazine.jsonl`)**: Cleaned text extracted from scanned monthly issues of Vipula, a popular literary publication containing short stories and essays.

### C. Encyclopedias \& Classical Literature
* **Vignana Sarvasvam (`telugu_vignana_sarvasvam.jsonl`)**: Entries from the Telugu Vignana Sarvasvam (encyclopedic dictionary), providing high-quality grammatical bases.
* **Puranas \& Classical Prose (`telugu_puranas_classical.jsonl`)**: Classical texts and epics translated into modern Telugu.

### D. Scanned Literature \& OCR
* **[Internet Archive Books](https://archive.org) (`telugu_ia_books.jsonl`)**: Cleaned OCR text extracted from public-domain Telugu books downloaded from the Internet Archive database.
* **Chitrakala OCR (`bharatiya_chitrakala_ocr.jsonl`)**: OCR text from art journals.
* **Navarasa Kadambari (`navarasa_kadambari_ocr.jsonl`)**: OCR text from literary novels.
* **Pranadanamu (`pranadanamu_ocr.jsonl`)**: OCR text from health/medical scanned manuals.

### E. Wikimedia Projects
* **[Telugu Wikipedia](https://te.wikipedia.org) (`telugu_wikipedia.jsonl`)**: Cleaned pages from the latest Telugu Wikipedia dump.
* **[Telugu Wikisource](https://te.wikisource.org) (`telugu_wikisource_manual.jsonl`)**: Scanned classical texts and transcriptions parsed from Wikisource.
* **[Telugu Wikibooks](https://te.wikibooks.org) (`telugu_wikibooks.jsonl`)**: Cleaned textbook data.
* **[Telugu Wikiquote](https://te.wikiquote.org) (`telugu_wikiquote.jsonl`)**: Curated quotations.
* **[Telugu Wiktionary](https://te.wiktionary.org) (`telugu_wiktionary.jsonl`)**: Telugu dictionary definitions.

---

## 2. Public Sources (302.70 Million Words Post-Cleaning)

These sources are sampled from large, publicly available Indic text dumps:

* **[IndicCorp (Telugu)](https://huggingface.co/datasets/ai4bharat/IndicCorp)**: A massive web crawl dataset developed by AI4Bharat, providing clean sentence blocks from Telugu online publications.
* **[CC-100 (Telugu)](https://data.statmt.org/cc-100/)**: The Telugu subset of the Common Crawl multilingual dataset compiled for XLM-R pretraining.
* **[Wikipedia (Telugu Dump)](https://dumps.wikimedia.org)**: Public parsed pages used to supplement encyclopedic terminology.
