# Nepali Dataset Collection Summary

## Overview
This document summarizes the dataset collection efforts for Nepali (our chosen low-resource language) for pre-training our language model. Despite being a low-resource language, we maximized data yield through programmatic API downloads and manual extraction (Wikipedia dumps, WikiSource dumps, and PDF OCR).

## Data Sources Used

### High-Volume Downloaded Sources
1. **CulturaX (Nepali)**: A massive web-crawled dataset, serving as the largest contributor to the corpus.
2. **Sangraha (Verified & Unverified)**: Specialized Nepali datasets.
3. **Nepali Wikipedia (HF Dataset)**: Pre-processed Wikipedia articles.

### Manually Collected Sources
1. **Nepali Wikipedia Dump (XML)**: Direct XML parsing of the latest encyclopedic articles.
2. **Nepali WikiSource Dump**: Primary source texts, including historical scripts and literature.
3. **Scanned PDFs (OCR)**: Scanned historical/classic Nepali novels and books (e.g., *Sumnima*) processed using Tesseract OCR with the `nep` language pack.

## Data Format
All raw data is standardized into JSON Lines (`.jsonl`) format. Each line is a valid JSON object:
```json
{
  "text": "The actual Nepali text content...",
  "source": "culturax_ne" // or "wikipedia", "ocr_sumnima", etc.
}
```

## Content Types
- **Web Crawl**: Large volumes of internet text (news, blogs, social discourse) from CulturaX.
- **Encyclopedic/Factual**: Structured articles from Wikipedia covering global and regional knowledge.
- **Literary/Historical**: Classic texts from WikiSource and manual OCR efforts, providing deep cultural context and high-quality vocabulary.

## Problems to be Cleaned (Phase 2)
The datasets are currently raw and will undergo the following cleaning steps:
1. **Unicode Normalization**: Ensuring consistent Devanagari script representation and removing hidden control characters.
2. **Deduplication**: Web crawls inherently contain massive duplication. We will use MinHash LSH to remove exact and near-duplicates to prevent data contamination.
3. **OCR Errors**: OCR output contains misrecognized characters and line-break issues that need heuristic filtering.
4. **Language Purity**: Some sources contain significant amounts of English or mixed scripts. We need to filter texts based on the ratio of Devanagari characters to ASCII characters.
5. **Formatting Noise**: Removing leftover wiki syntax, HTML tags, and short/low-quality lines.
