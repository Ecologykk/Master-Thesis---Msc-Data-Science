from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import asyncio
from crawl4ai import AsyncWebCrawler

def get_dgsi_acordao_links():
    """
    Use Selenium to get acordão links from DGSI search results in a fully automated way.
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Re-enable for automation
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(options=chrome_options)
    wait = WebDriverWait(driver, 20)  # Wait for up to 20 seconds
    
    try:
        print("🔍 Step 1: Navigating to the search form...")
        driver.get("https://www.dgsi.pt/jtre.nsf/Pesquisa+Livre?OpenForm")
        
        print("🔍 Step 2: Filling and submitting the search form...")
        # Wait for the input field to be present and locate it
        search_input = wait.until(EC.presence_of_element_located((By.NAME, "Query")))
        search_input.send_keys("VIOLÊNCIA DOMÉSTICA")
        
        # Find and click the submit button
        submit_btn = driver.find_element(By.CSS_SELECTOR, "input[type='submit']")
        submit_btn.click()
        
        print("🔍 Step 3: Waiting for search results to load...")
        # Wait for a known element on the results page, like the results table or a link
        wait.until(EC.presence_of_element_located((By.XPATH, "//a[contains(@href, 'OpenDocument')]")))
        print("✅ Search results loaded.")

        # Optional: Save page source for debugging if needed
        # with open("debug_page.html", "w", encoding="utf-8") as f:
        #     f.write(driver.page_source)
        
        print("🔍 Step 4: Extracting links from the results page...")
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        acordao_links = set() # Use a set to automatically handle duplicates
        all_links = soup.find_all('a', href=True)
        
        for link in all_links:
            href = link['href']
            # Be more specific to only get document links
            if 'OpenDocument' in href and '/jtre.nsf/' in href:
                if not href.startswith('http'):
                    href = 'https://www.dgsi.pt' + href
                acordao_links.add(href)
        
        print(f"✅ Found {len(acordao_links)} unique acordão links.")
        return list(acordao_links)
        
    except Exception as e:
        print(f"❌ An error occurred during link extraction: {e}")
        # Save a screenshot for debugging failures
        driver.save_screenshot("error_screenshot.png")
        print("📸 Screenshot saved to error_screenshot.png")
        return []
    finally:
        print("🚪 Closing browser.")
        driver.quit()

async def scrape_acordao_with_crawl4ai(url, semaphore, session_id):
    """
    Scrapes an individual acordão, controlled by a semaphore to limit concurrency.
    """
    # Wait for our turn to use a resource slot
    async with semaphore:
        print(f"[{session_id}/{num_to_scrape}] 🟢 Starting scrape for: {url}")
        async with AsyncWebCrawler(headless=True) as crawler:
            try:
                result = await crawler.arun(url=url, css_selector="body")
                
                if result.status_code == 200:
                    soup = BeautifulSoup(result.html, 'html.parser')
                    
                    # --- Metadata Extraction ---
                    metadata = {"url": url}
                    # Example: Extracting the "Processo" number (adjust selector as needed)
                    try:
                        processo_element = soup.find(lambda tag: 'Processo:' in tag.text)
                        if processo_element:
                            metadata['processo'] = processo_element.get_text(strip=True).replace('Processo:', '').strip()
                    except:
                        metadata['processo'] = None

                    # --- Content Cleaning ---
                    for element in soup(["script", "style", "nav", "header", "footer", "form"]):
                        element.decompose()
                    
                    text = soup.get_text(separator='\n', strip=True)
                    
                    print(f"[{session_id}/{num_to_scrape}] ✅ Success: {url}")
                    return {
                        "status": "success",
                        "metadata": metadata,
                        "content_text": text
                    }
                else:
                    print(f"[{session_id}/{num_to_scrape}] ⚠️ HTTP Error {result.status_code}: {url}")
                    return {"status": "error", "url": url, "details": f"HTTP status {result.status_code}"}
                    
            except Exception as e:
                print(f"[{session_id}/{num_to_scrape}] ❌ Exception: {url} - {e}")
                return {"status": "error", "url": url, "details": str(e)}

# These need to be global or passed around for the progress counter
num_to_scrape = 0

async def main():
    global num_to_scrape
    # Step 1: Get links
    acordao_links = get_dgsi_acordao_links()
    
    if not acordao_links:
        print("❌ No links found. Exiting.")
        return

    # Step 2: Scrape the documents with controlled concurrency
    num_to_scrape = len(acordao_links)
    # Set concurrency limit. 10 is a safe starting point. Adjust based on your machine's power.
    CONCURRENCY_LIMIT = 50
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    print(f"\nPhase 2: Scraping {num_to_scrape} acordãos with a concurrency of {CONCURRENCY_LIMIT}...")
    
    tasks = [scrape_acordao_with_crawl4ai(link, semaphore, i+1) for i, link in enumerate(acordao_links)]
    scraped_results = await asyncio.gather(*tasks)
    
    # Step 3: Process and save results
    successful_scrapes = [res for res in scraped_results if res and res['status'] == 'success']
    failed_scrapes = [res for res in scraped_results if res and res['status'] == 'error']
    
    print(f"\n✅ Scraping complete. Success: {len(successful_scrapes)}, Failures: {len(failed_scrapes)}")
    
    # Save all successful results to a single JSON file
    if successful_scrapes:
        import json
        output_filename = "dgsi_acordaos.json"
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(successful_scrapes, f, ensure_ascii=False, indent=4)
        print(f"💾 All data saved to {output_filename}")

    if failed_scrapes:
        print("⚠️ Some URLs failed to scrape. Check failed_urls.txt for details.")
        with open("failed_urls.txt", "w") as f:
            for failure in failed_scrapes:
                f.write(f"{failure['url']} - Reason: {failure['details']}\n")

if __name__ == "__main__":
    asyncio.run(main())