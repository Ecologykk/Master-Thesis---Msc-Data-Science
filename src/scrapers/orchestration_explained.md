
## Scraper orchestration script

### Dependencies



| Tool         | Docs Link                                              | Version                 | Motive                                                                                                                                                                                                                                                                                                                                                                                                                |
| ------------ | ------------------------------------------------------ | ----------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python       | https://www.python.org/downloads/release/python-31210/ | 3.12.10                 | Easy to work for this task, has all necessary tools.                                                                                                                                                                                                                                                                                                                                                                  |
| Asyncio      | https://docs.python.org/3/library/asyncio.html         | Built in Python version | Necessary for the multiple doc scraping, altough i scale down for sequential for more stability.                                                                                                                                                                                                                                                                                                                      |
| Datetime     | https://docs.python.org/3/library/datetime.html        | Built in Python version | Used for date and time handling. For this script was only for unique id for each file                                                                                                                                                                                                                                                                                                                                 |
| BeutifulSoup | https://www.crummy.com/software/BeautifulSoup/         | 4.13.5                  | Usef for HTML parsing and Document Object Model Trasversal (meaning efficiently find and manipulate HTML elements)                                                                                                                                                                                                                                                                                                    |
| Crawl4AI     | https://docs.crawl4ai.com/                             | 0.7.4                   | Modern Alternative that lets scrape pages asyncronously and extract the text in a markdown format, making it easier for ML/DL ingestion. It was used to obtain the url from the pages yet it was not fully used to its capabilites as it was not deemed worth it. "requests" would of have same performance for this case, but would not allow further performance upgrades in the future for massive volume scraping |
| Pathlib      | https://docs.python.org/3/library/pathlib.html         | Built in Python version | Modern library to deal with paths. Specially useful when dealing with multiple OS. Here was used just to store the final JSON file.                                                                                                                                                                                                                                                                                   |
| Selenium     | https://www.selenium.dev/documentation/                | 4.35.0                  | Used for browser automation, like filling forms etc. In my case i decided a more manual approach, because i prefered to actually search the documents myself to ensure i was right. In the future selenium could be used to fully atutomate the browser operations done.                                                                                                                                              |
| JSON         | https://docs.python.org/3/library/json.html            | Built in Python version | JSON file manipulator. Since JSON was my target output, this library was crucial.                                                                                                                                                                                                                                                                                                                                     |


### Code explaining

#### 1. Importing

```python
import asyncio

import json

import re

from datetime import datetime

from pathlib import Path

from bs4 import BeautifulSoup

from selenium import webdriver

from selenium.webdriver.chrome.options import Options

from domestic_violence_scraper import DomesticViolenceScraper

from contract_breach_scraper import ContractBreachScraper

```

- **asyncio** - Again this one was meant more for the concurrent approach, but kept it either way, it did not break the code. Also it was necessary to run craw4ai
- **datetime** - only really used to id files, not super important
- **re** - Regex (Regular expression library), necessary for textual pattern extraction. 
- **beutiful soup** - the html parser that enables the metadata extraction and full judgment decision extraction.
-  **crawl4ai** - this was first used as the concurrent scraper it basically used internally bs4 to parse the html and extract the text. Altough in our case in the more simple version i just used it simply to obtain the urls present in the webpage. In the sequential approach i took, libraries like "request" would of been just as good. However, craw4ai concurrent abilities allow for future re-work of the script.
- **json** - library used to create and manipulate the output json files
- **selenium** - this library is extremely powerful and allow us to simulate browser behaviours. At its peak it can automate fully browsing, but since im not really doing a deployment-level script and i want more reliability rather than speed i decided to open the browsers and manually search for what i wanted. In the future or another contributor could fully automate the searches.


#### 2.1 Initialization

```python
def __init__(self):

        """Initialize orchestrator with available scrapers and court URLs."""

        self.dv_scraper = DomesticViolenceScraper()

        self.cb_scraper = ContractBreachScraper()

        self.output_dir = Path("../data")

        self.output_dir.mkdir(exist_ok=True)

        # Court URLs for different tribunals

      self.court_urls_tr = {

            "TRE": "https://www.dgsi.pt/jtre.nsf/Pesquisa+Descritor?OpenForm", # TRE - Tribunal de Relação de Évora / Évora Court of Appeal

            "TRL": "https://www.dgsi.pt/jtrl.nsf/Pesquisa+Descritor?OpenForm", # TRL - Tribunal de Relação de Lisboa / Lisbon Court of Appeal

            "TRC": "https://www.dgsi.pt/jtrc.nsf/Pesquisa+Descritor?OpenForm", # TRC - Tribunal de Relação de Coimbra / Coimbra Court of Appeal

            "TRG": "https://www.dgsi.pt/jtrg.nsf/Pesquisa+Descritor?OpenForm", # TRG - Tribunal de Relação de Guimarães / Guimarães Court of Appeal

            "TRP": "https://www.dgsi.pt/jtrp.nsf/Pesquisa+Descritor?OpenForm", # TRP - Tribunal de Relação do Porto / Porto Court of Appeal

            "STJ": "https://www.dgsi.pt/jstj.nsf/Pesquisa+Descritor?OpenForm"  # STJ - Supremo Tribunal de Justiça / Supreme Court

        }

        self.court_url_jp = "https://www.dgsi.pt/cajp.nsf/Pesquisa+Campo?OpenForm"
```

Here we initialize the classes regarding the scraping and processing of domestic violence and breach of contract cases. 
We also initilalize the instances regarding the links for each court search page.

#### 2.1 Orchestration of scraping, trough the CLI

Now ideally this should not be as manual as i made it, but once again, i re-iterate my goal was not performance but reliability.

```python
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

            print(f"\n🔍 Opening DGSI search page...")

            driver.get(tribunal_url)

            print("\n✅ Interactive mode - MULTI-PAGE EXTRACTION")

            print("   🎯 This mode allows extracting links from multiple pages.\n")

            print("="*80)

            if search_type == "TR":

                print("⚠️  WORKFLOW FOR COURTS OF APPEAL (Domestic Violence):")

                print("="*80)

                print("1. Search by keywords (e.g., 'violência doméstica')")

                print("2. You'll see a list of possible DESCRIPTORS")

                print("3. ⚡ CLICK on a specific descriptor (e.g., 'VÍTIMA DE VIOLÊNCIA DOMÉSTICA')")

            else:  # JP

                print("⚠️  WORKFLOW FOR PEACE COURTS (Contract Breach):")

                print("="*80)

                print("1. Search by descriptors/fields (e.g., 'incumprimento de contrato')")

                print("2. You'll see a page with results")

            print("4. Now you'll see the LIST OF JUDGMENTS")

            print("5. On this results page, come back here and press '1' to extract")

            print("6. If there are multiple result pages, navigate between them and repeat")

            print("="*80 + "\n")

            while True:

                print(f"\n📊 Links accumulated so far: {len(all_links)}")

                print("\n🔄 Available options:")

                print("   1. Extract links from current page")

                print("   2. Finish and continue with scraping")

                print("   3. Exit without scraping")

                choice = input("\nChoose an option (1-3): ").strip()

                if choice == "1":

                    print("\n🔍 Extracting links from current page...")

                    soup = BeautifulSoup(driver.page_source, 'html.parser')

                    current_url = driver.current_url

                    print(f"🔍 Current browser URL: {current_url}")

                    page_links = set()

                    for link in soup.find_all('a', href=True):

                        href = link['href']

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

                        print("\n⚠️  No judgment links found on this page.")

                        print("💡 POSSIBLE CAUSES:")

                        print("   1. You're on the DESCRIPTOR LIST page (need to click a descriptor)")

                        print("   2. You're on the search page (need to search first)")

                        print("   3. Page is still loading (wait and try again)")

                    else:

                        print(f"\n✅ Links extracted successfully!")

                        if len(new_links) > 0:

                            print(f"📝 Examples of new links found:")

                            for i, link in enumerate(list(new_links)[:3]):

                                print(f"  {i+1}. {link[:100]}...")

                    print("\n👆 Navigate to the next page in the browser and come back here to extract more.")

                elif choice == "2":

                    if len(all_links) > 0:

                        print(f"\n✅ Finishing extraction with {len(all_links)} unique links.")

                        return list(all_links)

                    else:

                        print("❌ No links extracted. Exiting.")

                        return []

                elif choice == "3":

                    print("🚪 Exiting without scraping.")

                    return []

                else:

                    print("❌ Invalid option. Choose 1, 2, or 3.")

        except Exception as e:

            print(f"❌ Error during link extraction: {e}")

            return []

        finally:

            print("🚪 Closing search browser.")

            driver.quit()
```

Main CLI workflow, that uses the method of extracting links above:
```python

async def run(self):

        """Main orchestration workflow - simplified like test11.py."""

        print("=" * 80)

        print("🏛️  DGSI LEGAL DOCUMENT SCRAPER".center(80))

        print("=" * 80)

        print("\n📋 Choose case type:")

        print("   1. Courts of Appeal - Domestic Violence  (TRE, TRL, TRC, TRG, TRP, STJ)")

        print("   2. Peace Courts - Contract Breach")

        choice = input("\nChoose an option (1-2): ").strip()

        if choice == "1":

            # DOMESTIC VIOLENCE - COURTS OF APPEAL

            print("\n🏛️ Available Courts of Appeal:")

            print(f"   {', '.join(self.court_urls_tr.keys())}")

            print(f"TRE - Évora | TRL - Lisboa | TRC - Coimbra | TRG - Guimarães | TRP - Porto | STJ - Supremo  " )

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

                "STJ": "jstj"  # Supreme Court uses 'jstj' not 'jtrj'

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

                        print(f"⚠️  Requested ({num_to_scrape}) > total. Processing all ({total}).")

                        num_to_scrape = total

            except (ValueError, TypeError):

                print("⚠️  Invalid input. Processing first 5 by default.")

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

                        print(f"⚠️  Requested ({num_to_scrape}) > total. Processing all ({total}).")

                        num_to_scrape = total

            except (ValueError, TypeError):

                print("⚠️  Invalid input. Processing first 5 by default.")

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

        else:

            print("❌ Invalid option. Exiting.")

            return
```

I tried making it quite self explanatory if someone else wants to use it. An overview:

1. First it asks what's the search type the user wants (courts of appeal or courts of peace).[^1 Note I considered the Supreme Court, an Appeal Court even though technically it's not. Just for simplicity I did.]

2. If the user chooses courts of appeal, it must then choose which district(e.g Lisboa, Porto, etc). If it choose courts of peace it moves to the next step as there is no disitiction per locality.

3. Then the browser opens in the correspondent descriptor search page. Afterwards the user must search either "violencia domestica" or "incumprimento contratual". 

4. A new page is loaded with the correspondent descriptors and related ones. This is good to catch as many cases of the given types. Then it must press the ones he wants to scrape. 

5. As he goes to the next page with all cases possible (with a max limit of 500 judgments, something that limits this study) then he presses "1" if he wants to scrape those urls. The browser remains open, so it can go back and press other descriptors and repeat the process said.

6. When it gets all links he wants, then he presses "2", to scrape the list of links he accumulated. "3" allows to break out of the script.

7. Then when "2" is employed we just need to wait for scraping to be done.



#### 2.2 Sequentially scraping the documents

```python

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

                    print(f"[{i}/{len(links)}] ⚠️  Failed to extract metadata")

            except Exception as e:

                failed_urls.append(url)

                print(f"[{i}/{len(links)}] ❌ Exception: {str(e)[:60]}")

            # Pause between requests

            await asyncio.sleep(1)
```





Simple acessory method, allowing to iterate over the list of extracted links, then one by one scraping the relevant information from the page html, using the method *scrape_judgment()* from either the Domestic Violence Scraper or Breach of Contract Scraper.

Final Note:
---
The script used was sequential, intentionally since parallel scraping using crawl4ai, was not giving much performance improvements. Seems plausible since i was scraping only a few hundred of documents at a time. It seems the concurrent approach could be more useful in a later study when scraping tens or even hundreds of thousands of documents at a time.

Here is a snippet on how to implement concurrent scraping using crawl4ai, with a semaphore limiters, meaning the maximum of documents it will scrape at the same time. This number is critical and there is a global optimum associated with it. Too high it might eat too many computer resources, slowing down the process. Too low and we might not see the actual concurrent processing abilities.

1. **Modify this orchestration script [main.py](vscode-file://vscode-app/c:/Users/helto/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)**, replace [scrape_documents()](vscode-file://vscode-app/c:/Users/helto/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) with:
```python
async def scrape_documents_parallel(self, links, scraper, court_name, 
                                   case_type, max_concurrent=5):
    """Parallel scraping with concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def limited_scrape(url, session_id):
        async with semaphore:
            try:
                doc = await scraper.scrape_judgment(url, court_name, case_type)
                await asyncio.sleep(0.5)  # Polite delay
                return doc
            except Exception as e:
                print(f"[{session_id}] Error: {e}")
                return None
    
    tasks = [limited_scrape(url, i+1) for i, url in enumerate(links)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    documents = [r for r in results if r is not None]
    failed = [links[i] for i, r in enumerate(results) if r is None]
    
    return documents, failed
```


Notice that parallel processing might bring some errors at times, that happen under the hood (which honestly I couldn't even understand, let alone explain to you clearly). So the safest and most reliable option is sequential. 