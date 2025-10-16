# Master Thesis - MSc Data Science

**Can LLMs predict court cases decisions?**

This repository contains the code and data for a Master's thesis in Data Science focused on applying Large Language Models (LLMs) to Portuguese legal document analysis.

## 🎯 Research Objectives

1. **Outcome Prediction**: Can LLMs predict and justify court case outcomes based on judgment text?


## 📁 Project Structure

```
Master-Thesis---Msc-Data-Science/
├── src/
│   └── scrapers/                    # Legal document scrapers
│       ├── domestic_violence_scraper.py
│       ├── contract_breach_scraper.py
│       ├── main.py                  # Interactive CLI
│       ├── README.md                # Detailed documentation
│       ├── orchestration_explained.md   # Main.py deep-dive
│       └── scrapers_explained.md        # Scraper algorithms deep-dive
│
├── data/                            # Scraped documents (JSON)
│
└── README.md                        # This file
```

## 🚀 Current Status: Data Collection Phase

### What's Working
✅ **Domestic Violence Scraper**: Collecting cases from Courts of Appeal (TRE, TRL, TRC, TRG, TRP) and Supreme Court (STJ)  
✅ **Contract Breach Scraper**: Collecting cases from Peace Courts (JP)  
✅ **Decision Extraction Algorithm**: 15-layer cascade to separate final decisions from judgment text (prevents data leakage)  
✅ **Interactive CLI**: User-friendly terminal interface for scraping

### What's Not Started
⏳ Data preprocessing and cleaning  
⏳ Data Exploration
⏳ ML/DL baseline modelling
⏳ LLM model training and evaluation  
⏳ LLM improvement with RAG - (if possible)
⏳ Comparative analysis between case types
⏳ Legal experts validation


## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install beautifulsoup4 crawl4ai  selenium
```

### 2. Run Scrapers

```bash
cd src/scrapers
python main.py
```

Follow the interactive prompts to:
1. Choose case type (Domestic Violence or Contract Breach)
2. Select court
3. Navigate to results page (Selenium opens browser)
4. Extract links and scrape documents automatically

## 🔬 Key Feature: Decision Extraction

**Why separate the decision from judgment text?**
- Prevents **data leakage** in ML models
- The decision section contains the verdict (our prediction target)
- Including it in training features would leak the answer
- Forces models to learn from legal reasoning, not outcomes

**15-layer cascade algorithm**:
- Searches for explicit section markers (Roman numerals, letters)
- Identifies concluding expressions ("Pelo exposto", "Acordam")
- Adaptive threshold (searches last 30% → 40% → 50% of document)
- Prioritizes semantic patterns over isolated numbering (avoids false positives)

## 📊 Data Collected So Far

| Case Type | Source | Courts | Documents |
|-----------|--------|--------|-----------|
| **Domestic Violence** | Courts of Appeal | TRE, TRL, TRC, TRG, TRP, STJ | Thousands |
| **Contract Breach** | Peace Courts | Various JP | Hundreds |

## � Output Schema

```json
{
  "url": "https://www.dgsi.pt/...",
  "tribunal": "TRE",
  "tipo_caso": "VIOLÊNCIA DOMÉSTICA",
  "n_processo": "123/45.6ABCDE",
  "juiz_relator": "Judge Name",
  "data_acordao": "01-01-2024",
  "descritores": ["Descriptor 1", "Descriptor 2"],
  "sumario": "Summary text...",
  "texto_integral_completo": "Full judgment text...",
  "decisao_extraida_do_texto_integral": "Decision only...",
  "texto_integral_sem_decisao": "Text without decision...",
  "metadata_decisao": {
    "extraction_method": "dispositivo_roman",
    "confidence": "high",
    "requires_manual_review": false
  }
}
```

## � Documentation

- **Scraper Details**: See `src/scrapers/README.md`
- **Algorithm Explanation**: Docstrings in scraper modules

## 👤 Author

**Helton Mendonça**
Master's Student in Data Science @ Universidade de Lisboa - Faculdade de Ciências (LASIGE)  
Thesis: Comparing LLM performance on ethically charged vs. documental legal cases

## 🙏 Acknowledgments
- **Thesis Advisors** for guidance and support

---

**Status**: � Data Collection Phase  
**Last Updated**: October 2025
