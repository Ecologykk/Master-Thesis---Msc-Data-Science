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
    Use Selenium to get acordão links from DGSI search results
    """
    # Setup Chrome options - remove headless to see what's happening
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Commented for debugging
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        print("🔍 Step 1: Go to search form page...")
        driver.get("https://www.dgsi.pt/jtre.nsf/Pesquisa+Livre?OpenForm")
        time.sleep(3)
        
        print("🔍 Step 2: Fill search form...")
        
        # Find search input field
        search_inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='text'], textarea")
        print(f"Found {len(search_inputs)} input elements")
        
        search_found = False
        for i, inp in enumerate(search_inputs):
            try:
                if inp.is_displayed() and inp.is_enabled():
                    print(f"Using input {i}: name={inp.get_attribute('name')}")
                    inp.clear()
                    inp.send_keys("VIOLÊNCIA DOMÉSTICA")
                    
                    # Find submit button
                    submit_btn = driver.find_element(By.CSS_SELECTOR, "input[type='submit'], button[type='submit']")
                    print("Submitting search...")
                    submit_btn.click()
                    search_found = True
                    break
                    
            except Exception as e:
                print(f"Input {i} failed: {e}")
                continue
        
        # Wait for manual control - you can browse and navigate manually
        print("🔍 Step 3: Manual control phase...")
        print("Navigate to the results page manually if needed.")
        print("Press Enter when you're ready to extract links from the current page...")
        input()
        
        # Debug: save page source
        with open("debug_page.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        print("Saved page source to debug_page.html")
        
        # Extract links with better URL handling
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        acordao_links = []
        all_links = soup.find_all('a', href=True)
        print(f"Total links found: {len(all_links)}")
        
        for link in all_links:
            href = link['href']
            if 'OpenDocument' in href or '.nsf/' in href:
                # Fix URL construction to avoid truncation
                if not href.startswith('http'):
                    # Handle relative URLs properly
                    if href.startswith('/'):
                        href = 'https://www.dgsi.pt' + href
                    else:
                        href = 'https://www.dgsi.pt/' + href
                
                # Ensure the full URL is preserved
                full_url = href
                acordao_links.append(full_url)
                print(f"Added link: {full_url}")
        
        print(f"✅ Found {len(acordao_links)} potential acordão links")
        
        # Save links to file for inspection
        with open("extracted_links.txt", "w", encoding="utf-8") as f:
            for i, link in enumerate(acordao_links, 1):
                f.write(f"{i:03d}: {link}\n")
        print("Saved all links to extracted_links.txt")
        
        return acordao_links
        
    finally:
        print("Browser will stay open until you press Enter...")
        input("Press Enter to close browser...")
        driver.quit()

async def scrape_acordao_with_crawl4ai(url):
    """
    Use Crawl4AI to scrape individual acordão and convert to clean markdown
    """
    async with AsyncWebCrawler(headless=True) as crawler:
        try:
            print(f"Attempting to scrape: {url}")
            result = await crawler.arun(
                url=url,
                delay_before_return_html=3,  # Increased delay
                css_selector="body"
            )
            
            if result.status_code == 200:
                # Clean up the HTML content
                soup = BeautifulSoup(result.html, 'html.parser')
                
                # Remove unwanted elements
                for element in soup(["script", "style", "nav", "header", "footer"]):
                    element.decompose()
                
                # Get clean text
                text = soup.get_text(separator='\n')
                
                # Clean whitespace
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                clean_text = '\n\n'.join(lines)
                
                # Create markdown
                markdown = f"""# Acordão

**URL:** {url}

**Status:** Scraped successfully

---

{clean_text}

---
"""
                return markdown
            else:
                error_msg = f"# ERRO\n\nStatus Code: {result.status_code}\nFalha ao carregar: {url}\n"
                print(f"❌ HTTP {result.status_code}: {url}")
                return error_msg
                
        except Exception as e:
            error_msg = f"# ERRO\n\nErro: {str(e)}\nURL: {url}\n"
            print(f"❌ Exception: {e}")
            return error_msg

async def main():
    # Step 1: Get links with Selenium
    print("Phase 1: Extracting acordão links with Selenium...")
    acordao_links = get_dgsi_acordao_links()
    
    if not acordao_links:
        print("❌ No links found")
        return
    
    # Step 2: Test first link to check for URL issues
    print(f"\nPhase 2: Testing first link...")
    if acordao_links:
        test_markdown = await scrape_acordao_with_crawl4ai(acordao_links[0])
        
        with open("test_acordao.md", 'w', encoding='utf-8') as f:
            f.write(test_markdown)
        print("✅ Saved test result to test_acordao.md")
        
        # Check if test was successful
        if "ERRO" not in test_markdown:
            print("✅ Test successful! URLs appear to be working.")
            
            # Ask user if they want to continue with more
            response = input(f"\nFound {len(acordao_links)} total links. How many do you want to scrape? (Enter number or 'all'): ")
            
            if response.lower() == 'all':
                num_to_scrape = len(acordao_links)
            else:
                try:
                    num_to_scrape = min(int(response), len(acordao_links))
                except ValueError:
                    num_to_scrape = 5
            
            print(f"\nPhase 3: Scraping {num_to_scrape} acordãos...")
            
            for i, link in enumerate(acordao_links[:num_to_scrape]):
                print(f"Scraping {i+1}/{num_to_scrape}: {link[:80]}...")
                markdown = await scrape_acordao_with_crawl4ai(link)
                
                # Save to file
                filename = f"acordao_{i+1:03d}.md"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(markdown)
                
                print(f"✅ Saved: {filename}")
                
                # Small delay between requests
                await asyncio.sleep(1)
        
        else:
            print("❌ Test failed. Check the URLs in extracted_links.txt for issues.")
    
    print(f"\n✅ Complete! Found {len(acordao_links)} total links")

if __name__ == "__main__":
    asyncio.run(main())