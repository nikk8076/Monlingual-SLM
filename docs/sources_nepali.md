# Nepali Monolingual Dataset — List of Sources

This document lists all manual and public data sources aggregated to construct the 400M-word cleaned Nepali monolingual corpus.

---

## 1. Manual Sources (100.08 Million Words Post-Cleaning)

These sources represent high-quality regional data collected manually via API scraping, web scraping, and OCR parsing.

### A. Scraped News Portals
* **[Onlinekhabar](https://www.onlinekhabar.com) (`nepali_news_onlinekhabar_huge.jsonl`)**: Expanded manually scraped articles extracted via the [WordPress REST API](https://www.onlinekhabar.com/wp-json/wp/v2/posts) endpoint to resolve duplicate pagination routing issues on the website.
* **[Ekantipur](https://ekantipur.com) (`nepali_news_ekantipur.jsonl`)**: Crawled articles from the Ekantipur news website covering various categories (News, Sports, Business, Entertainment).
* **[Setopati](https://setopati.com) (`nepali_news_setopati.jsonl`)**: Crawled articles from Setopati portal.
* **[Ratopati](https://ratopati.com) (`nepali_news_ratopati.jsonl`)**: Crawled articles from Ratopati portal.
* **16 News Portals (`nepali_16news.jsonl`)**: A unified crawl across 16 local and national Nepali news outlets.
* **[Gorkhapatra](https://gorkhapatraonline.com) (`nepali_gorkhapatra.jsonl`)**: Scraped articles from Gorkhapatra, the oldest state-owned news publication in Nepal.

### B. Scanned Literature \& OCR
* **Classical Books (`nepali_classical_books.jsonl`)**: Cleaned OCR text extracted from historical Nepali literature, classical prose, and novels (e.g., Sumnima).
* **[Internet Archive Books](https://archive.org) (`nepali_ia_books.jsonl`)**: OCR texts extracted from public-domain Nepali books downloaded from the Internet Archive database.
* **Historical Magazines (`nepali_magazine_corpus.jsonl` / `nepali_purnima_magazine.jsonl`)**: Transcripts of historical journals, local magazines, and cultural periodicals.

### C. Reference Dictionaries \& Religious Texts
* **Brihat Sabdakosh (`nepali_brihat_sabdakosh.jsonl`)**: Complete entries from the Nepali Brihat Sabdakosh (comprehensive dictionary), providing rich morphological bases for tokenization.
* **Vedic \& Puranic Corpus (`nepali_all_vedas.jsonl`, `nepali_skanda_purana.jsonl`, `nepali_bhagwat_mahapuran2.jsonl`)**: Sanskrit religious texts translated into classical Nepali.

### D. Wikimedia Projects
* **[Wikipedia Manual Dump](https://dumps.wikimedia.org) (`nepali_wikipedia_manual.jsonl`)**: Cleaned text extracted from the latest dump of the Nepali Wikipedia.
* **[Wikibooks](https://dumps.wikimedia.org) (`nepali_wikibooks.jsonl`)**: Manual extraction of textbook content.
* **[Wiktionary](https://dumps.wikimedia.org) (`nepali_wiktionary.jsonl`)**: Dictionary terms and lexical meanings.

---

## 2. Public Sources (299.92 Million Words Post-Cleaning)

These sources are sampled from large, publicly available Indic text dumps:

* **[IndicCorp (Nepali)](https://huggingface.co/datasets/ai4bharat/IndicCorp)**: A massive web crawl dataset developed by AI4Bharat, providing clean sentence blocks from online publications.
* **[CC-100 (Nepali)](https://data.statmt.org/cc-100/)**: The Nepali subset of the Common Crawl multilingual dataset compiled for XLM-R pretraining.
* **[Wikipedia (Nepali Dump)](https://dumps.wikimedia.org)**: Public parsed pages used to supplement encyclopedic terminology.
