# Legal Document Scrapers

See [`docs/02-scraping.md`](../../docs/02-scraping.md) for setup, CLI usage, output schema, and
the November 2025 corpus-snapshot warning.

The decision-separation algorithm (progressive 15-layer cascade, used to prevent ML data
leakage) is documented on `_extract_decision_from_text` in
[`domestic_violence_scraper.py`](domestic_violence_scraper.py) and
[`contract_breach_scraper.py`](contract_breach_scraper.py).
