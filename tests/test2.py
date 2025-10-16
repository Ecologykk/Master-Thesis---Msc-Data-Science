import asyncio
from crawl4ai import AsyncWebCrawler
import time

async def scrape_dgsi_search_results():
    """
    Scrape DGSI search results with proper dynamic content handling
    """
    
    # Your original URL
    search_url = "https://www.dgsi.pt/jtre.nsf/8f8d2b7e72244bca80256879006d6594?CreateDocument"
    
    async with AsyncWebCrawler(
        # Critical: Enable JavaScript execution
        headless=True,
        browser_type="chromium",  # Use Chromium for better JS support
        verbose=True
    ) as crawler:
        
        try:
            result = await crawler.arun(
                url=search_url,
                
                # Wait for content to load
                wait_for_selector="table",  # Wait for results table
                delay_before_return_html=5,  # Wait 5 seconds after page load
                
                # JavaScript execution settings
                js_code=[
                    # Wait for any loading indicators to disappear
                    "await new Promise(resolve => setTimeout(resolve, 3000));",
                    
                    # Scroll to trigger any lazy loading
                    "window.scrollTo(0, document.body.scrollHeight);",
                    
                    # Wait a bit more
                    "await new Promise(resolve => setTimeout(resolve, 2000));",
                ],
                
                # CSS selector to wait for (the search results)
                css_selector=".search-results, table, .documento",
                
                # Additional wait conditions
                wait_for_js=True,
                
                # Session persistence
                session_id="dgsi_session"
            )
            
            print(f"Status Code: {result.status_code}")
            print(f"Content Length: {len(result.html)}")
            print(f"URL after crawling: {result.url}")
            
            # Check if we got actual content
            if "500 documents found" in result.html or "documentos encontrados" in result.html:
                print("✅ Successfully found search results!")
                return result.html
            else:
                print("❌ No search results found in HTML")
                print("First 1000 characters:")
                print(result.html[:1000])
                return None
                
        except Exception as e:
            print(f"Error during crawling: {e}")
            return None

async def alternative_approach_with_selenium():
    """
    Alternative approach using selenium-like behavior
    """
    
    async with AsyncWebCrawler(
        headless=False,  # Set to True for production
        browser_type="chromium",
        verbose=True
    ) as crawler:
        
        # First, try to navigate to the main search page
        main_url = "https://www.dgsi.pt/jtre.nsf/"
        
        result = await crawler.arun(
            url=main_url,
            js_code=[
                # Try to find and fill search form
                """
                // Look for search input field
                const searchInput = document.querySelector('input[type="text"]') || 
                                   document.querySelector('input[name*="search"]') ||
                                   document.querySelector('input[name*="query"]');
                
                if (searchInput) {
                    searchInput.value = 'VIOLÊNCIA DOMÉSTICA';
                    
                    // Find and click search button
                    const searchButton = document.querySelector('input[type="submit"]') ||
                                        document.querySelector('button[type="submit"]') ||
                                        document.querySelector('.search-button');
                    
                    if (searchButton) {
                        searchButton.click();
                        
                        // Wait for results to load
                        await new Promise(resolve => setTimeout(resolve, 5000));
                    }
                }
                """
            ],
            wait_for_selector="table",
            delay_before_return_html=8
        )
        
        return result.html

async def scrape_individual_acordaos(html_content):
    """
    Extract individual acordão links from the search results
    """
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Look for links to individual documents
    acordao_links = []
    
    # Common patterns for acordão links in DGSI
    link_patterns = [
        'a[href*=".nsf"]',  # Lotus Notes database links
        'a[href*="acordao"]',  # Direct acordão links
        'a[href*="OpenDocument"]'  # Document opening links
    ]
    
    for pattern in link_patterns:
        links = soup.select(pattern)
        for link in links:
            href = link.get('href')
            if href and 'OpenDocument' in href:
                # Make sure it's a full URL
                if not href.startswith('http'):
                    href = 'https://www.dgsi.pt' + href
                acordao_links.append(href)
    
    return acordao_links

# Main execution
async def main():
    print("🔍 Attempting to scrape DGSI search results...")
    
    # Try the first approach
    html_content = await scrape_dgsi_search_results()
    
    if not html_content:
        print("🔄 First approach failed, trying alternative...")
        html_content2 = await alternative_approach_with_selenium()
    
    if html_content2:
        print("📄 Extracting individual acordão links...")
        acordao_links = await scrape_individual_acordaos(html_content2)
        
        print(f"Found {len(acordao_links)} acordão links:")
        for i, link in enumerate(acordao_links[:10]):  # Show first 10
            print(f"{i+1}. {link}")
            
        return acordao_links
    else:
        print("❌ Failed to retrieve search results")
        return []

# Run the scraper
if __name__ == "__main__":
    links = asyncio.run(main())