"""
Legal Document Scrapers Package

This package provides specialized scrapers for extracting legal documents from
Portuguese courts, avaiable publicly in dgsi.pt.

Modules:
- domestic_violence_scraper: Scrapes domestic violence cases from Courts of Appeal
- contract_breach_scraper: Scrapes contract breach cases from Peace Courts
- main: Orchestration script for interactive scraping

Author: Helton Mendonça

Transparency Note: This code was optimized and polished using AI assistance(mainly Claude 4.0 and 4.5). However, all the logic, structure and final review were done by the author.

Date: October 13th 2025
"""

from .domestic_violence_scraper import DomesticViolenceScraper
from .contract_breach_scraper import ContractBreachScraper

__all__ = ['DomesticViolenceScraper', 'ContractBreachScraper']
__version__ = '1.0.0'
