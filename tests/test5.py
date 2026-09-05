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
        search_inputs = driver.find_elements(
            By.CSS_SELECTOR, "input[type='text'], textarea"
        )
        print(f"Found {len(search_inputs)} input elements")

        search_found = False
        for i, inp in enumerate(search_inputs):
            try:
                if inp.is_displayed() and inp.is_enabled():
                    print(f"Using input {i}: name={inp.get_attribute('name')}")
                    inp.clear()
                    inp.send_keys("VIOLÊNCIA DOMÉSTICA")

                    # Find submit button
                    submit_btn = driver.find_element(
                        By.CSS_SELECTOR, "input[type='submit'], button[type='submit']"
                    )
                    print("Submitting search...")
                    submit_btn.click()
                    search_found = True
                    break

            except Exception as e:
                print(f"Input {i} failed: {e}")
                continue

        # Wait for results to load
        print("🔍 Step 3: Waiting for results...")
        time.sleep(10)

        # Debug: save page source
        with open("debug_page.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        print("Saved page source to debug_page.html")

        # Extract links
        soup = BeautifulSoup(driver.page_source, "html.parser")

        acordao_links = []
        all_links = soup.find_all("a", href=True)
        print(f"Total links found: {len(all_links)}")

        for link in all_links:
            href = link["href"]
            if "OpenDocument" in href or ".nsf/" in href:
                if not href.startswith("http"):
                    href = "https://www.dgsi.pt" + href
                acordao_links.append(href)
                print(f"Added link: {href[:100]}...")

        print(f"✅ Found {len(acordao_links)} potential acordão links")
        return acordao_links

    finally:
        input("Press Enter to close browser...")  # Keep browser open for debugging
        driver.quit()


async def scrape_acordao_with_crawl4ai(url):
    """
    Use Crawl4AI to scrape individual acordão and convert to clean markdown
    """
    async with AsyncWebCrawler(headless=True) as crawler:
        try:
            result = await crawler.arun(
                url=url, delay_before_return_html=2, css_selector="body"
            )

            if result.status_code == 200:
                # Clean up the HTML content
                soup = BeautifulSoup(result.html, "html.parser")

                # Remove unwanted elements
                for element in soup(["script", "style", "nav", "header", "footer"]):
                    element.decompose()

                # Get clean text
                text = soup.get_text(separator="\n")

                # Clean whitespace
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                clean_text = "\n\n".join(lines)

                # Create markdown
                markdown = f"""# Acordão

**URL:** {url}

---

{clean_text}

---
"""
                return markdown
            else:
                return f"# ERRO\n\nFalha ao carregar: {url}\n"

        except Exception as e:
            return f"# ERRO\n\nErro: {str(e)}\nURL: {url}\n"


async def main():
    # Step 1: Get links with Selenium
    print("Phase 1: Extracting acordão links with Selenium...")
    acordao_links = get_dgsi_acordao_links()

    if not acordao_links:
        print("❌ No links found")
        return

    # Step 2: Scrape first few with Crawl4AI
    print(f"\nPhase 2: Scraping first 5 acordãos with Crawl4AI...")

    for i, link in enumerate(acordao_links[:5]):
        print(f"Scraping {i+1}/5: {link}")
        markdown = await scrape_acordao_with_crawl4ai(link)

        # Save to file
        filename = f"acordao_{i+1:03d}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(markdown)

        print(f"✅ Saved: {filename}")

    print(f"\n✅ Complete! Found {len(acordao_links)} total links, scraped first 5")
    print("To scrape all, increase the slice from [:5] to [:len(acordao_links)]")


if __name__ == "__main__":
    asyncio.run(main())
