import asyncio
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup

async def debug_dgsi_form():
    """
    First, let's see what the actual form looks like
    """
    async with AsyncWebCrawler(headless=False, verbose=True) as crawler:
        
        # Debug JavaScript to inspect the form
        debug_js = """
        console.log('=== DEBUGGING DGSI FORM ===');
        
        // Find all forms
        const forms = document.querySelectorAll('form');
        console.log('Forms found:', forms.length);
        
        forms.forEach((form, i) => {
            console.log(`Form ${i}:`, form.action, form.method);
            
            // Find inputs in this form
            const inputs = form.querySelectorAll('input, textarea, select');
            inputs.forEach((input, j) => {
                console.log(`  Input ${j}:`, {
                    type: input.type,
                    name: input.name,
                    id: input.id,
                    placeholder: input.placeholder,
                    value: input.value
                });
            });
        });
        
        // Also check for any search-related elements
        const searchElements = document.querySelectorAll('[name*="search"], [id*="search"], [class*="search"]');
        console.log('Search-related elements:', searchElements.length);
        
        searchElements.forEach((el, i) => {
            console.log(`Search element ${i}:`, {
                tag: el.tagName,
                type: el.type,
                name: el.name,
                id: el.id,
                className: el.className
            });
        });
        
        return 'Debug complete';
        """
        
        result = await crawler.arun(
            url="https://www.dgsi.pt/jtre.nsf/",
            js_code=[debug_js],
            delay_before_return_html=3
        )
        
        return result.html

async def try_direct_search_url():
    """
    Let's try to construct the search URL directly
    """
    async with AsyncWebCrawler(headless=True, verbose=True) as crawler:
        
        # Try common search URL patterns
        search_urls = [
            "https://www.dgsi.pt/jtre.nsf/SearchView?SearchView&Query=VIOLÊNCIA+DOMÉSTICA",
            "https://www.dgsi.pt/jtre.nsf/frmSearch?SearchView",
            "https://www.dgsi.pt/jtre.nsf/search?query=VIOLÊNCIA+DOMÉSTICA",
        ]
        
        for url in search_urls:
            print(f"Trying: {url}")
            try:
                result = await crawler.arun(
                    url=url,
                    delay_before_return_html=3
                )
                
                if "OpenDocument" in result.html:
                    print(f"✅ Found results at: {url}")
                    
                    soup = BeautifulSoup(result.html, 'html.parser')
                    links = [a['href'] for a in soup.find_all('a', href=True) 
                            if 'OpenDocument' in a['href']]
                    
                    print(f"Found {len(links)} acordão links")
                    return links
                    
            except Exception as e:
                print(f"Failed: {e}")
                continue
        
        return []

async def analyze_original_url():
    """
    Let's carefully analyze your original working URL
    """
    original_url = "https://www.dgsi.pt/jtre.nsf/8f8d2b7e72244bca80256879006d6594?CreateDocument"
    
    async with AsyncWebCrawler(
        headless=False,  # Let's see what happens
        verbose=True
    ) as crawler:
        
        result = await crawler.arun(
            url=original_url,
            delay_before_return_html=8,
            wait_for_js=True
        )
        
        print(f"Original URL status: {result.status_code}")
        print(f"Final URL: {result.url}")
        
        # Check if it redirects or loads content
        soup = BeautifulSoup(result.html, 'html.parser')
        
        # Look for any forms or search interfaces
        forms = soup.find_all('form')
        print(f"Forms found: {len(forms)}")
        
        # Look for the search results we expect
        if "500 documents found" in result.html or "documentos" in result.html:
            print("✅ Original URL does work!")
            links = [a['href'] for a in soup.find_all('a', href=True) 
                    if 'OpenDocument' in a['href']]
            return links
        
        return []

async def main():
    print("🔍 Step 1: Debug form structure")
    await debug_dgsi_form()
    
    print("\n🔍 Step 2: Try direct search URLs")  
    links = await try_direct_search_url()
    
    if not links:
        print("\n🔍 Step 3: Analyze original URL")
        links = await analyze_original_url()
    
    if links:
        print(f"\n✅ Success! Found {len(links)} links")
        for i, link in enumerate(links[:5]):
            print(f"{i+1}. {link}")
    else:
        print("\n❌ No links found")

if __name__ == "__main__":
    asyncio.run(main())