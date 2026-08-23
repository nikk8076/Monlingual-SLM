# Telugu Dataset Collection Summary

## Overview
This document summarizes the dataset collection efforts for Telugu (a high-resource language) for pre-training our language model. The collection is divided into programmatic API downloads (high-volume) and manual data extraction (Wikipedia dumps, WikiSource dumps, and PDF OCR).

## Data Sources Used

### High-Volume Downloaded Sources
1. **CulturaX (Telugu)**: Massive multilingual dataset.
2. **Sangraha (Verified & Unverified)**: Culturally relevant datasets in Telugu.
3. **Telugu Books (Kaggle/HF)**: Contemporary fiction and non-fiction texts.

### Manually Collected Sources
1. **Telugu Wikipedia Dump**: Encyclopedic articles.
2. **Telugu WikiSource Dump**: Primary source texts, historical documents, poems, and scriptures.
3. **Scanned PDFs (OCR)**: Selected classic/historical books (e.g., *Pranadanamu*, *Navarasa Kadambari*, *Bharatiya Chitrakala*) processed using Tesseract OCR.

## Data Format
All raw data has been standardized into JSON Lines (`.jsonl`) format. Each line is a valid JSON object containing at minimum:
```json
{
  "text": "The actual Telugu text content...",
  "source": "culturax_te" // or "wikipedia", "sangraha", "ocr_pranadanamu", etc.
}
```

## Content Types
- **Encyclopedic/Factual**: Wikipedia articles containing structured knowledge, historical facts, and scientific information.
- **Literary/Historical**: Wikisource dumps containing classical literature, poetry, and historical archives.
- **Web Crawl / Conversational**: CulturaX and Sangraha datasets comprising news articles, blogs, forums, and general web text.
- **Digitized Books**: High-quality OCR outputs of culturally significant Telugu books covering philosophy, art, and storytelling.

## Problems to be Cleaned (Phase 2)
The current datasets are completely raw and exhibit several issues that need to be addressed before training:
1. **Unicode Inconsistencies**: Variations in Unicode representations (NFC vs NFD) and presence of stray control characters.
2. **Duplication**: High likelihood of near-duplicates (especially across web crawls like CulturaX) which need to be removed using MinHash LSH.
3. **OCR Noise**: The Tesseract OCR outputs contain garbled characters, broken lines, and misrecognized words.
4. **Script Mixing**: Presence of English/ASCII characters or numbers mixed heavily with Telugu script, requiring language-ratio filtering.
5. **Formatting Artifacts**: Extraneous HTML tags, wiki markup, or PDF headers/footers left over from extraction.
