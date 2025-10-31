"""
Legal Document Scraper - Orchestration Script

This script provides an interactive command-line interface for scraping legal documents
from Portuguese courts using specialized scrapers for different case types. It was developed to perform research 
on Master's thesis in Data Science at the University of Lisbon, to perform predictions on legal outcomes using Machine Learning.

Supported Case Types:
- Domestic Violence (from Courts of Appeal - Tribunais de Relação)
- Contract Breach (from Peace Courts - Julgados de Paz)
- Contract Breach (from Courts of Appeal - Tribunais de Relação) 🔄
- Contract Breach (from Supreme Court - STJ) 🔄
- Contract Breach (from CSM Database - jurisprudencia.csm.org.pt) 🔄

Features:
- Interactive link extraction with Selenium (user navigates, script extracts). This allowed multi-page extraction without too much complexity of form automation, although that could be improved in the future.
- Batch document scraping with progress tracking. It's sequential because the parallel version had many issues with timeouts and reliability, so I kept it simple for now. This also could be improved in the future.
- JSON export with metadata. It extracts more metadata that it needs for the thesis, but could be useful for other research purposes in the future.
- Automatic separation of decision text to prevent ML data leakage. This part still could be improved in the future, but its quite robust as of the current version. 
There is quite a lot of variability in how decisions are formatted, because judges write them in different ways, making it hard to cover all cases.

Author: Helton Mendonça

Transparency Note: This code was optimized and polished using AI assistance(mainly Claude 4.0 and 4.5). However, all the logic, structure and final review were done by the author.

Date: October 13th 2025
"""

import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from domestic_violence_scraper import DomesticViolenceScraper
from contract_breach_scraper import ContractBreachScraper


class LegalDocumentOrchestrator:
    """
    Orchestrates the scraping process for different types of legal documents.
    Simplified workflow: interactive link extraction -> user selects quantity -> sequential scraping.
    """
    
    def __init__(self):
        """Initialize orchestrator with available scrapers and court URLs."""
        self.dv_scraper = DomesticViolenceScraper()
        self.cb_scraper = ContractBreachScraper()
        self.output_dir = Path("../../data")
        self.output_dir.mkdir(exist_ok=True)
        
        # Court URLs for different tribunals
        self.court_urls_tr = {
            "TRE": "https://www.dgsi.pt/jtre.nsf/Pesquisa+Descritor?OpenForm", # TRE - Tribunal de Relação de Évora / Évora Court of Appeal
            "TRL": "https://www.dgsi.pt/jtrl.nsf/Pesquisa+Descritor?OpenForm", # TRL - Tribunal de Relação de Lisboa / Lisbon Court of Appeal
            "TRC": "https://www.dgsi.pt/jtrc.nsf/Pesquisa+Descritor?OpenForm", # TRC - Tribunal de Relação de Coimbra / Coimbra Court of Appeal
            "TRG": "https://www.dgsi.pt/jtrg.nsf/Pesquisa+Descritor?OpenForm", # TRG - Tribunal de Relação de Guimarães / Guimarães Court of Appeal
            "TRP": "https://www.dgsi.pt/jtrp.nsf/Pesquisa+Descritor?OpenForm", # TRP - Tribunal de Relação do Porto / Porto Court of Appeal
            "STJ": "https://www.dgsi.pt/jstj.nsf/Pesquisa+Descritor?OpenForm"  # STJ - Supremo Tribunal de Justiça / Supreme Court
        }
        self.court_url_jp = "https://www.dgsi.pt/cajp.nsf/Pesquisa+Campo?OpenForm"
        self.court_url_csm = "https://jurisprudencia.csm.org.pt/"
    
    def extract_links_interactive(self, search_type, tribunal_url, tribunal_id):
        """
        Extract judgment links interactively using Selenium.
        User navigates the DGSI website, script extracts links from current page.
        
        Args:
            search_type: "TR" (Tribunais de Relação) or "JP" (Julgados de Paz)
            tribunal_url: URL of the search form
            tribunal_id: Tribunal identifier (e.g., "jtrl", "cajp")
            
        Returns:
            list: Unique judgment URLs
        """
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        driver = webdriver.Chrome(options=chrome_options)
        all_links = set()
        
        try:
            if tribunal_id == "csm":
                print(f"\n🔍 Opening CSM database page...")
            else:
                print(f"\n🔍 Opening DGSI search page...")
            driver.get(tribunal_url)
            
            print("\n✅ Interactive mode - MULTI-PAGE EXTRACTION")
            print("   🎯 This mode allows extracting links from multiple pages.\n")
            print("="*80)
            
            if search_type == "TR":
                print("⚠️  WORKFLOW FOR COURTS OF APPEAL:")
                print("="*80)
                print("1. Search by keywords (e.g., 'violência doméstica', 'incumprimento de contrato')")
                print("2. You'll see a list of possible DESCRIPTORS")
                print("3. ⚡ CLICK on a specific descriptor (e.g., 'VÍTIMA DE VIOLÊNCIA DOMÉSTICA')")
            elif search_type == "JP":
                print("⚠️  WORKFLOW FOR PEACE COURTS (Contract Breach):")
                print("="*80)
                print("1. Search by descriptors/fields (e.g., 'incumprimento de contrato')")
                print("2. You'll see a page with results")
            else:  # CSM
                print("⚠️  WORKFLOW FOR CSM DATABASE:")
                print("="*80)
                print("1. Search manually (e.g., 'incumprimento de contrato')")
                print("2. Navigate through result pages")
                print("3. ⚠️  Note: Different link structure - may require adaptation")
            
            print("4. Now you'll see the LIST OF JUDGMENTS")
            print("5. On this results page, come back here and press '1' to extract")
            print("6. If there are multiple result pages, navigate between them and repeat")
            print("="*80 + "\n")
            
            while True:
                print(f"\n📊 Links accumulated so far: {len(all_links)}")
                print("\n🔄 Available options:")
                print("   1. Extract links from current page")
                print("   2. Finish and continue with scraping")
                print("   3. Exit without scraping")
                if tribunal_id == "csm":
                    print("   4. 🚀 Auto-paginate through ALL CSM pages (recommended for CSM)")
                
                if tribunal_id == "csm":
                    choice = input("\nChoose an option (1-4): ").strip()
                else:
                    choice = input("\nChoose an option (1-3): ").strip()
                
                if choice == "1":
                    print("\n🔍 Extracting links from current page...")
                    
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    current_url = driver.current_url
                    print(f"🔍 Current browser URL: {current_url}")
                    
                    page_links = set()
                    
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        
                        if tribunal_id == "csm":
                            # CSM website specific pattern - ECLI links
                            if 'ecli-item-anchor' in link.get('class', []):
                                if href.startswith('/ecli/'):
                                    if not href.startswith('http'):
                                        href = 'https://jurisprudencia.csm.org.pt' + href
                                    page_links.add(href)
                        else:
                            # DGSI website patterns (existing logic)
                            if f'{tribunal_id}.nsf' in href.lower():
                                is_valid = False
                                
                                # Valid link patterns
                                if 'OpenDocument' in href:
                                    is_valid = True
                                elif re.search(r'/[a-f0-9]{32}/', href, re.IGNORECASE):
                                    is_valid = True
                                elif re.match(rf'.*{tribunal_id}\.nsf/[a-f0-9]{{32}}', href, re.IGNORECASE):
                                    is_valid = True
                                
                                # Exclude navigation/search links
                                if any(x in href for x in ['SearchView', 'CreateDocument', 'OpenForm', 'Pesquisa']):
                                    is_valid = False
                                
                                if is_valid:
                                    if not href.startswith('http'):
                                        href = 'https://www.dgsi.pt' + href
                                    page_links.add(href)
                    
                    new_links = page_links - all_links
                    all_links.update(page_links)
                    
                    print(f"\n✅ Extracted {len(page_links)} links from this page")
                    print(f"📈 New unique links: {len(new_links)}")
                    print(f"📊 Total accumulated: {len(all_links)} unique links")
                    
                    if len(page_links) == 0:
                        print("\n⚠️  No judgment links found on this page.")
                        print("💡 POSSIBLE CAUSES:")
                        print("   1. You're on the DESCRIPTOR LIST page (need to click a descriptor)")
                        print("   2. You're on the search page (need to search first)")
                        print("   3. Page is still loading (wait and try again)")
                    else:
                        print(f"\n✅ Links extracted successfully!")
                        if len(new_links) > 0:
                            print(f"📝 Examples of new links found:")
                            for i, link in enumerate(list(new_links)[:3]):
                                print(f"  {i+1}. {link[:100]}...")
                    
                    print("\n👆 Navigate to the next page in the browser and come back here to extract more.")
                    
                elif choice == "2":
                    if len(all_links) > 0:
                        print(f"\n✅ Finishing extraction with {len(all_links)} unique links.")
                        return list(all_links)
                    else:
                        print("❌ No links extracted. Exiting.")
                        return []
                
                elif choice == "4" and tribunal_id == "csm":
                    # NEW: Auto-paginate through CSM results
                    print("\n🔄 Auto-paginating through all CSM pages...")
                    current_page = 1
                    
                    # Dynamically find max pages from pagination element
                    try:
                        soup = BeautifulSoup(driver.page_source, 'html.parser')
                        # Look for pagination info - adjust selector based on actual CSM HTML structure
                        pagination = soup.find('span', class_='dynatable-page-links')
                        if pagination:
                            page_links = pagination.find_all('a', class_='dynatable-page-link')
                            max_pages = max([int(link.text) for link in page_links if link.text.isdigit()], default=1)
                        else:
                            # Fallback: look for total results and calculate
                            total_results = soup.find('span', class_='dynatable-record-count')
                            if total_results:
                                total = int(total_results.text)
                                max_pages = (total + 9) // 10  # Assuming 10 results per page
                            else:
                                max_pages = 200  # Conservative fallback
                        print(f"   📊 Detected {max_pages} total pages")
                    except Exception as e:
                        print(f"   ⚠️  Could not detect max pages: {e}. Using fallback of 100.")
                        max_pages = 200
                    
                    while current_page <= max_pages:
                        print(f"\n📄 Processing page {current_page}/{max_pages}...")
                        
                        # Extract links from current page
                        soup = BeautifulSoup(driver.page_source, 'html.parser')
                        page_links = set()
                        
                        for link in soup.find_all('a', href=True):
                            href = link['href']
                            if 'ecli-item-anchor' in link.get('class', []):
                                if href.startswith('/ecli/'):
                                    if not href.startswith('http'):
                                        href = 'https://jurisprudencia.csm.org.pt' + href
                                    page_links.add(href)
                        
                        new_links = page_links - all_links
                        all_links.update(page_links)
                        
                        print(f"   ✅ Extracted {len(page_links)} links from page {current_page}")
                        print(f"   📈 New unique links: {len(new_links)}")
                        print(f"   📊 Total accumulated: {len(all_links)} unique links")
                        
                        # Check if there's a "Próximo" (Next) button and click it
                        try:
                            # Wait a bit for any AJAX to settle
                            time.sleep(1)
                            
                            # Re-find the next button to avoid stale element reference
                            next_buttons = driver.find_elements("css selector", "a.dynatable-page-next")
                            next_button = None
                            
                            # Find the enabled "Próximo" button
                            for btn in next_buttons:
                                if "Próximo" in btn.text and "dynatable-disabled-page" not in btn.get_attribute("class"):
                                    next_button = btn
                                    break
                            
                            if next_button:
                                print(f"   🔄 Clicking 'Próximo' to go to page {current_page + 1}...")
                                
                                # Use JavaScript click to avoid element interception issues
                                driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                                time.sleep(0.5)
                                driver.execute_script("arguments[0].click();", next_button)
                                
                                # Wait for page to load and pagination to update
                                time.sleep(4)
                                current_page += 1
                            else:
                                print(f"   ⚠️  No enabled 'Próximo' button found. Reached end at page {current_page}.")
                                break
                                
                        except Exception as e:
                            print(f"   ⚠️  Could not find/click 'Próximo' button: {str(e)}")
                            print(f"   🔄 Attempting to continue anyway...")
                            
                            # Try one more time with a different approach
                            try:
                                time.sleep(2)
                                driver.execute_script("document.querySelector('a.dynatable-page-next:not(.dynatable-disabled-page)').click();")
                                time.sleep(4)
                                current_page += 1
                                print(f"   ✅ Successfully clicked via JavaScript")
                            except Exception as e2:
                                print(f"   ❌ JavaScript click also failed: {str(e2)}")
                                break
                    
                    print(f"\n🎉 Auto-pagination complete! Total pages processed: {current_page}")
                    print(f"🎯 Final count: {len(all_links)} unique links")
                    
                    if len(all_links) > 0:
                        return list(all_links)
                    else:
                        print("❌ No links extracted from auto-pagination.")
                        return []
                        
                elif choice == "3":
                    print("🚪 Exiting without scraping.")
                    return []
                else:
                    if tribunal_id == "csm":
                        print("❌ Invalid option. Choose 1, 2, 3, or 4.")
                    else:
                        print("❌ Invalid option. Choose 1, 2, or 3.")
            
        except Exception as e:
            print(f"❌ Error during link extraction: {e}")
            return []
        finally:
            print("🚪 Closing search browser.")
            driver.quit()
    
    async def scrape_documents(self, links, scraper, court_name, case_type):
        """
        Scrape documents sequentially from list of URLs.
        
        Args:
            links: List of judgment URLs
            scraper: DomesticViolenceScraper or ContractBreachScraper instance
            court_name: Court identifier
            case_type: Type of case
            
        Returns:
            tuple: (successful_documents, failed_urls)
        """
        documents = []
        failed_urls = []
        
        print(f"\n🚀 Scraping {len(links)} documents sequentially...")
        
        for i, url in enumerate(links, 1):
            print(f"\n[{i}/{len(links)}] 🟢 Processing: {url[:80]}...")
            
            try:
                doc = await scraper.scrape_judgment(url, court_name, case_type)
                
                if doc:
                    documents.append(doc)
                    print(f"[{i}/{len(links)}] ✅ Success! Process: {doc.get('n_processo', 'N/A')}")
                else:
                    failed_urls.append(url)
                    print(f"[{i}/{len(links)}] ⚠️  Failed to extract metadata")
                    
            except Exception as e:
                failed_urls.append(url)
                print(f"[{i}/{len(links)}] ❌ Exception: {str(e)[:60]}")
            
            # Pause between requests
            await asyncio.sleep(1)
        
        return documents, failed_urls
    
    def save_results(self, documents, filename):
        """Save scraped documents to JSON file."""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Results saved to: {filepath}")
    
    def save_failed_urls(self, failed_urls, filename="failed_urls.txt"):
        """Save failed URLs to text file."""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for url in failed_urls:
                f.write(url + '\n')
        
        print(f"⚠️  Failed URLs saved to: {filepath}")
    
    async def run(self):
        """Main orchestration workflow - simplified like test11.py."""
        print("=" * 80)
        print("🏛️  DGSI LEGAL DOCUMENT SCRAPER".center(80))
        print("=" * 80)
        print("\n📋 Choose case type:")
        print("   1. Courts of Appeal - Domestic Violence  (TRE, TRL, TRC, TRG, TRP, STJ)")
        print("   2. Peace Courts - Contract Breach")
        print("   3. 🔄 Courts of Appeal - Contract Breach  (TRE, TRL, TRC, TRG, TRP)")
        print("   4. 🔄 Supreme Court - Contract Breach  (STJ)")
        print("   5. 🔄 CSM Database - Contract Breach  (jurisprudencia.csm.org.pt)")
        print("   6. 🔄 CSM Database - Domestic Violence  (jurisprudencia.csm.org.pt)")
        
        choice = input("\nChoose an option (1-6): ").strip()
        
        if choice == "1":
            # DOMESTIC VIOLENCE - COURTS OF APPEAL
            print("\n🏛️ Available Courts of Appeal:")
            print(f"   {', '.join(self.court_urls_tr.keys())}")
            print(f"TRE - Évora | TRL - Lisboa | TRC - Coimbra | TRG - Guimarães | TRP - Porto | STJ - Supremo  " )
            
            court_code = input(f"\nChoose court: ").strip().upper()
            
            if court_code not in self.court_urls_tr:
                print("❌ Invalid court. Exiting.")
                return
            
            url = self.court_urls_tr[court_code]
            
            # Map court codes to DGSI tribunal identifiers
            tribunal_id_map = {
                "TRE": "jtre",
                "TRL": "jtrl",
                "TRC": "jtrc",
                "TRG": "jtrg",
                "TRP": "jtrp",
                "STJ": "jstj"  # Supreme Court uses 'jstj' not 'jtrj'
            }
            tribunal_id = tribunal_id_map.get(court_code, f'jtr{court_code[-1].lower()}')
            
            print(f"\n🎯 Mode: Court of Appeal - {court_code}")
            print(f"📝 Opening: {url}")
            print("💡 Topic: Domestic Violence")
            
            # Interactive link extraction
            links = self.extract_links_interactive("TR", url, tribunal_id)
            
            if not links:
                print("❌ No links extracted. Exiting.")
                return
            
            # Ask how many to process
            total = len(links)
            response = input(f"\n📊 Found {total} links. How many to process? (number or 'all'): ").strip()
            
            try:
                if response.lower() == 'all':
                    num_to_scrape = total
                else:
                    num_to_scrape = int(response)
                    if num_to_scrape > total:
                        print(f"⚠️  Requested ({num_to_scrape}) > total. Processing all ({total}).")
                        num_to_scrape = total
            except (ValueError, TypeError):
                print("⚠️  Invalid input. Processing first 5 by default.")
                num_to_scrape = 5
            
            links_to_scrape = links[:num_to_scrape]
            
            # Scrape documents
            documents, failed = await self.scrape_documents(
                links_to_scrape,
                self.dv_scraper,
                court_code,
                "VIOLÊNCIA DOMÉSTICA"
            )
            
            # Results summary
            print("\n" + "=" * 80)
            print(f"✅ SCRAPING COMPLETE".center(80))
            print("=" * 80)
            print(f"✅ Successes: {len(documents)}")
            print(f"❌ Failures: {len(failed)}")
            
            # Save results
            if documents:
                filename = f"dgsi_violencia_domestica_{court_code.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.save_results(documents, filename)
            
            if failed:
                self.save_failed_urls(failed)
        
        elif choice == "2":
            # CONTRACT BREACH - PEACE COURTS
            url = self.court_url_jp
            tribunal_id = "cajp"
            
            print(f"\n🎯 Mode: Peace Courts")
            print(f"📝 Opening: {url}")
            print("💡 Topic: Contract Breach")
            
            # Interactive link extraction
            links = self.extract_links_interactive("JP", url, tribunal_id)
            
            if not links:
                print("❌ No links extracted. Exiting.")
                return
            
            # Ask how many to process
            total = len(links)
            response = input(f"\n📊 Found {total} links. How many to process? (number or 'all'): ").strip()
            
            try:
                if response.lower() == 'all':
                    num_to_scrape = total
                else:
                    num_to_scrape = int(response)
                    if num_to_scrape > total:
                        print(f"⚠️  Requested ({num_to_scrape}) > total. Processing all ({total}).")
                        num_to_scrape = total
            except (ValueError, TypeError):
                print("⚠️  Invalid input. Processing first 5 by default.")
                num_to_scrape = 5
            
            links_to_scrape = links[:num_to_scrape]
            
            # Scrape documents
            documents, failed = await self.scrape_documents(
                links_to_scrape,
                self.cb_scraper,
                "JP",
                "INCUMPRIMENTO DE CONTRATOS"
            )
            
            # Results summary
            print("\n" + "=" * 80)
            print(f"✅ SCRAPING COMPLETE".center(80))
            print("=" * 80)
            print(f"✅ Successes: {len(documents)}")
            print(f"❌ Failures: {len(failed)}")
            
            # Save results
            if documents:
                filename = f"dgsi_incumprimento_contratos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.save_results(documents, filename)
            
            if failed:
                self.save_failed_urls(failed)
        
        elif choice == "3":
            # CONTRACT BREACH - APPEALS COURTS 🔄
            print("\n🏛️ Available Courts of Appeal for Contract Breach:")
            print(f"   {', '.join([k for k in self.court_urls_tr.keys() if k != 'STJ'])}")
            print(f"TRE - Évora | TRL - Lisboa | TRC - Coimbra | TRG - Guimarães | TRP - Porto" )
            
            court_code = input(f"\nChoose court: ").strip().upper()
            
            if court_code not in self.court_urls_tr or court_code == 'STJ':
                print("❌ Invalid court or use option 4 for STJ. Exiting.")
                return
            
            url = self.court_urls_tr[court_code]
            tribunal_id_map = {
                "TRE": "jtre", "TRL": "jtrl", "TRC": "jtrc", 
                "TRG": "jtrg", "TRP": "jtrp"
            }
            tribunal_id = tribunal_id_map[court_code]
            
            print(f"\n🎯 Mode: Court of Appeal - {court_code}")
            print(f"📝 Opening: {url}")
            print("💡 Topic: Contract Breach (Appeal Level)")
            
            links = self.extract_links_interactive("TR", url, tribunal_id)
            if not links:
                print("❌ No links extracted. Exiting.")
                return
            
            total = len(links)
            response = input(f"\n📊 Found {total} links. How many to process? (number or 'all'): ").strip()
            
            try:
                if response.lower() == 'all':
                    num_to_scrape = total
                else:
                    num_to_scrape = int(response)
                    if num_to_scrape > total:
                        print(f"⚠️  Requested ({num_to_scrape}) > total. Processing all ({total}).")
                        num_to_scrape = total
            except (ValueError, TypeError):
                print("⚠️  Invalid input. Processing first 5 by default.")
                num_to_scrape = 5
            
            links_to_scrape = links[:num_to_scrape]
            
            documents, failed = await self.scrape_documents(
                links_to_scrape,
                self.cb_scraper,
                court_code,
                "INCUMPRIMENTO DE CONTRATOS"
            )
            
            print("\n" + "=" * 80)
            print(f"✅ SCRAPING COMPLETE".center(80))
            print("=" * 80)
            print(f"✅ Successes: {len(documents)}")
            print(f"❌ Failures: {len(failed)}")
            
            if documents:
                filename = f"dgsi_incumprimento_contratos_{court_code.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.save_results(documents, filename)
            
            if failed:
                self.save_failed_urls(failed)
        
        elif choice == "4":
            # CONTRACT BREACH - SUPREME COURT 🔄
            court_code = "STJ"
            url = self.court_urls_tr[court_code]
            tribunal_id = "jstj"
            
            print(f"\n🎯 Mode: Supreme Court - {court_code}")
            print(f"📝 Opening: {url}")
            print("💡 Topic: Contract Breach (Supreme Court)")
            
            links = self.extract_links_interactive("TR", url, tribunal_id)
            if not links:
                print("❌ No links extracted. Exiting.")
                return
            
            total = len(links)
            response = input(f"\n📊 Found {total} links. How many to process? (number or 'all'): ").strip()
            
            try:
                if response.lower() == 'all':
                    num_to_scrape = total
                else:
                    num_to_scrape = int(response)
                    if num_to_scrape > total:
                        print(f"⚠️  Requested ({num_to_scrape}) > total. Processing all ({total}).")
                        num_to_scrape = total
            except (ValueError, TypeError):
                print("⚠️  Invalid input. Processing first 5 by default.")
                num_to_scrape = 5
            
            links_to_scrape = links[:num_to_scrape]
            
            documents, failed = await self.scrape_documents(
                links_to_scrape,
                self.cb_scraper,
                court_code,
                "INCUMPRIMENTO DE CONTRATOS"
            )
            
            print("\n" + "=" * 80)
            print(f"✅ SCRAPING COMPLETE".center(80))
            print("=" * 80)
            print(f"✅ Successes: {len(documents)}")
            print(f"❌ Failures: {len(failed)}")
            
            if documents:
                filename = f"dgsi_incumprimento_contratos_stj_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.save_results(documents, filename)
            
            if failed:
                self.save_failed_urls(failed)
        
        elif choice == "5":
            # CONTRACT BREACH - CSM DATABASE 🔄
            url = self.court_url_csm
            tribunal_id = "csm"
            
            print(f"\n🎯 Mode: CSM Database")
            print(f"📝 Opening: {url}")
            print("💡 Topic: Contract Breach (CSM Jurisprudence Database)")
            print("💡 Note: May contain duplicates with DGSI - check URLs in post-processing")
            
            links = self.extract_links_interactive("CSM", url, tribunal_id)
            if not links:
                print("❌ No links extracted. Exiting.")
                return
            
            total = len(links)
            response = input(f"\n📊 Found {total} links. How many to process? (number or 'all'): ").strip()
            
            try:
                if response.lower() == 'all':
                    num_to_scrape = total
                else:
                    num_to_scrape = int(response)
                    if num_to_scrape > total:
                        print(f"⚠️  Requested ({num_to_scrape}) > total. Processing all ({total}).")
                        num_to_scrape = total
            except (ValueError, TypeError):
                print("⚠️  Invalid input. Processing first 5 by default.")
                num_to_scrape = 5
            
            links_to_scrape = links[:num_to_scrape]
            
            documents, failed = await self.scrape_documents(
                links_to_scrape,
                self.cb_scraper,
                "CSM",
                "INCUMPRIMENTO DE CONTRATOS"
            )
            
            print("\n" + "=" * 80)
            print(f"✅ SCRAPING COMPLETE".center(80))
            print("=" * 80)
            print(f"✅ Successes: {len(documents)}")
            print(f"❌ Failures: {len(failed)}")
            
            if documents:
                filename = f"csm_incumprimento_contratos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.save_results(documents, filename)
            
            if failed:
                self.save_failed_urls(failed)
        
        elif choice == "6":
            # DOMESTIC VIOLENCE - CSM DATABASE 🔄
            url = self.court_url_csm
            tribunal_id = "csm"
            
            print(f"\n🎯 Mode: CSM Database")
            print(f"📝 Opening: {url}")
            print("💡 Topic: Domestic Violence (CSM Jurisprudence Database)")
            print("💡 Note: May contain duplicates with DGSI - check URLs in post-processing")
            
            links = self.extract_links_interactive("CSM", url, tribunal_id)
            if not links:
                print("❌ No links extracted. Exiting.")
                return
            
            total = len(links)
            response = input(f"\n📊 Found {total} links. How many to process? (number or 'all'): ").strip()
            
            try:
                if response.lower() == 'all':
                    num_to_scrape = total
                else:
                    num_to_scrape = int(response)
                    if num_to_scrape > total:
                        print(f"⚠️  Requested ({num_to_scrape}) > total. Processing all ({total}).")
                        num_to_scrape = total
            except (ValueError, TypeError):
                print("⚠️  Invalid input. Processing first 5 by default.")
                num_to_scrape = 5
            
            links_to_scrape = links[:num_to_scrape]
            
            documents, failed = await self.scrape_documents(
                links_to_scrape,
                self.dv_scraper,
                "CSM",
                "VIOLÊNCIA DOMÉSTICA"
            )
            
            print("\n" + "=" * 80)
            print(f"✅ SCRAPING COMPLETE".center(80))
            print("=" * 80)
            print(f"✅ Successes: {len(documents)}")
            print(f"❌ Failures: {len(failed)}")
            
            if documents:
                filename = f"csm_violencia_domestica_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.save_results(documents, filename)
            
            if failed:
                self.save_failed_urls(failed)
        
        else:
            print("❌ Invalid option. Exiting.")
            return


async def main():
    """Entry point for the scraper."""
    orchestrator = LegalDocumentOrchestrator()
    await orchestrator.run()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("  DGSI Legal Document Scraper - Portuguese Courts")
    print("  Domestic Violence & Contract Breach Cases")
    print("  Author: Helton Mendonça ")
    print("="*80 + "\n")
    
    asyncio.run(main())
