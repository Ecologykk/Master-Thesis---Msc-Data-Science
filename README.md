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
│   └── data_processing_exploration/ # Data cleaning, processing, EDA
│       ├── processing_tests.ipynb
│       └── processing_auxiliary/
│           ├── fix_csm.json.py
│           ├── fix_csm_text.py
│           ├── fix_ic_json.py
│           └── pron_cases_solve.py
├                           
│   
│       
│       
│   
└── README.md                        # This file
```

## 🚦 Data Processing & EDA Status

### Data Scale & Integrity

- **Total scraped:** 6,627 documents (4,368 contract breach, 2,258 domestic violence)
- **Post-cleaning:** 1,938 usable contract breach cases (after dropping incomplete, duplicates, unusable)
- **Integrity filtering:**
  - ~44% of contract breach cases lacked full text (expected for DGSI)
  - ~5.5% had full text but missing decision (237 problematic cases, dropped). 4.4% for domestic violence corpus (equivalant to 99 cases, all dropped)
  - Final dataset: robust subset where `texto_integral` and `decisao` coexist

### Decision Extraction & Label Harmonization

- **Extraction:** Initial regex caught majority of decisions; manual-assist function built but skipped (marginal gain, high cost in time)
- **Summary creation:** Regex pattern mining from verbose texts; missing summaries filled intelligently, preserving metadata
- **Label mapping:** Collapsed legal expressions into interpretable ML labels

#### For Contract Breach:

| Setup      | Favorable | Unfavorable | Partial | Notes                |
|------------|-----------|-------------|---------|----------------------|
| Binary     | 54%       | 46%         | -       | Balanced for ML      |
| Ternary    | 36%       | 43%         | 21%     | Slight imbalance     |

#### For Domestic Violence:

| Setup      | Favorable | Unfavorable | Partial | Notes                |
|------------|-----------|-------------|---------|----------------------|
| Binary     | 80%       | 20%         | -       | Imbalanced for ML    |
| Ternary    | 63%       | 14%         | 23%     | Imbalanced imbalance     |

- **Legal logic:** Linguistic/procedural mapping matches appellate reasoning:
  - Favorable: *(Recurso)* Provido or Procedente, *(Sentença Anterior)* Revogada
  - Unfavorable: *(Recurso)* Improcedente or Negado Provimento, *(Sentença Anterior)* Confirmada
  - Partial: Parcialmente + action term

### EDA Highlights

- **Basic stats:** Distribution of dates, text length, word/sentence counts, length per class, class distribution, n-gram frequency per class (stopwords removed)
- **Descriptors:** Frequency analysis outside main ones (e.g., 'VIOLÊNCIA DOMÉSTICA', 'INCUMPRIMENTO DO/DE CONTRATO/CONTRATUAL')
- **Judge gender:** Names mapped to gender (male, female, non-descriptive); class distribution by gender
- **Tribunal:** Class distribution by court
- **Sentiment analysis:** Compared emotional tone between domestic violence and contract breach cases
- **Readability:** Flesch Reading Ease scores compared between case types

### Results Summary

| Step                | Contract Breach (IC) | Domestic Violence (DV) | Notes                                  |
|---------------------|---------------------|------------------------|----------------------------------------|
| Scraped             | 4,368               | 2,258                  |                                        |
| Usable (final)      | 1,874               | 1,138                  | DV had 100% full texts; IC ~44% missing|
| Dropped (no decision,with full text)| 5.46%              | 4.41%                  |                                        |
| Label Distribution  | See above           | See above              | Binary & ternary setups                |
| Readability         | Compared            | Compared               | ARI and Coleman Liau Scores adapted to Portuguese (both hard to read)                       |
| Sentiment           | Compared            | Compared               |   Both have negative sentiment. Domestic Violence more sentimental though                                    |
| Judge Gender        | Mapped              | Mapped                 |   No gender bias found                                     |
| Tribunal            | Analyzed            | Analyzed               |     No major tribunal bias, but further research could find interesting patterns                              |

**Summary:**  
- Data cleaning and integrity filtering produced a high-quality, robust dataset for ML.
- Decision extraction and label harmonization followed legal logic and practical judicial analytics.
- EDA covered statistical, linguistic, and legal dimensions, supporting downstream modeling.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install beautifulsoup4 crawl4ai selenium
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

| Case Type           | Source             | Courts                        | Documents |
|---------------------|--------------------|-------------------------------|-----------|
| Domestic Violence   | Courts of Appeal   | TRE, TRL, TRC, TRG, TRP, STJ  | Thousands |
| Contract Breach     | Peace Courts       | Various JP                    | Thousands  |

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

**Status**: � Data Collection & Processing Phase  
**Last Updated**: November 2025