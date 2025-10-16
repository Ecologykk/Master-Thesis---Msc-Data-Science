# Legal Document Scrapers

This package provides specialized scrapers for extracting legal documents from Portuguese courts.

## 📁 Structure

```
/src/scrapers/
              ├── __init__.py                      # Package initialization
              ├── domestic_violence_scraper.py     # Scraper for domestic violence cases
              ├── contract_breach_scraper.py       # Scraper for contract breach cases
              ├── main.py                          # Orchestration script (CLI)
              └── README.md                        # This file
```

## 🎯 Features

### Domestic Violence Scraper
- **Source**: Courts of Appeal (Tribunais de Relação) and the Supreme Court of Justice
- **Courts**: TRE, TRL, TRC, TRG, TRP, STJ
- **Case Type**: Violência Doméstica (Domestic Violence)
- **Extracted Fields**:
  - Process number, judge, date, descriptors, voting
  - Summary, full text, and **separated final decision**
  - Metadata about decision extraction method and confidence

### Contract Breach Scraper
- **Source**: Peace Courts (Julgados de Paz)
- **Courts**: Various JP
- **Case Type**: Incumprimento de Contratos (Contract Breach)
- **Extracted Fields**:
  - Process number, judge, date, descriptors, area
  - Summary, full text, and **separated final decision**
  - Metadata about decision extraction method and confidence

### Decision Extraction Algorithm
Both scrapers use the same robust **15-layer cascade algorithm** to extract the final decision (dispositivo) from judgment text:

1. **Layer 1-2**: Explicit section markers (Roman numerals III-X, Letters A-F)
2. **Layer 2.5**: Numbered sections with decision keywords
3. **Layer 3-4**: Concluding expressions ("Pelo exposto", "Acordam")
4. **Layer 5**: Isolated numbering (low priority to avoid false positives)
5. **Layer 6-7**: Adaptive threshold and edge cases

**Why separate the decision?**
- Prevents **data leakage** in ML models
- The decision section contains the verdict, which is often the prediction target
- Including it in training data would give models access to the answer they should predict

## 🚀 Usage

### Interactive CLI 

```bash
cd scrapers
python main.py
```

Follow the menu prompts to:
1. Select case type (Domestic Violence or Contract Breach)
2. Choose court and scraping parameters
3. Wait for extraction and save results to `data/` directory



## 📊 Output Format

Each scraped document is a dictionary with the following structure:

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

## 🔧 Dependencies

```bash
pip install beautifulsoup4 crawl4ai 
```

## 📝 Notes

- **Decision Extraction Confidence Levels**:
  - `high`: Explicit section markers found
  - `medium-high`: Strong indicators found
  - `medium`: Moderate confidence
  - `medium-low`: Weak indicators (manual review recommended)
  - `low`: No pattern found

- **Manual Review Flag**: Some documents are flagged with `requires_manual_review: true` when:
  - Isolated decimal numbering is used (may be fact lists)
  - Atypical document structure
  - Very short or long decision sections

- **Adaptive Threshold**: The algorithm searches in the last 30% of the document first, then expands to 40% and 50% if no pattern is found.

## 🤝 Contributing

This package is part of a Master's thesis in Data Science focusing on legal outcome prediction, comparing ethically charged cases like domestic violence against more documental and less ethical charged cases like breach of contracts.

Improvements and bug reports are welcome.

## 📄 License

Academic Research Project @ Faculdade de Ciências de Lisboa in LASIGE - 2025/2026



