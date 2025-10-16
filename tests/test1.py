
# Basic one page crawl test

# import asyncio
# from crawl4ai import AsyncWebCrawler

# async def main():
#     async with AsyncWebCrawler() as crawler:
#         result = await crawler.arun("https://www.dgsi.pt/jtre.nsf/134973db04f39bf2802579bf005f080b/ea4c7823fef4d56580258cf80039c931?OpenDocument&Highlight=0,VIOL%C3%8ANCIA,DOM%C3%89STICA")
#         print(result.markdown)  # Print first 300 chars

# if __name__ == "__main__":
#     asyncio.run(main())


# # Deep multi-page crawl test- > Only for static HTML. This website requires JS commands, to search keyworkds so it finds the correct data


# import asyncio
# from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
# from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
# from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy

# async def main():
#     # Configure a 2-level deep crawl
#     config = CrawlerRunConfig(
#         deep_crawl_strategy=BFSDeepCrawlStrategy(
#             max_depth=2, 
#             include_external=False
#         ),
#         scraping_strategy=LXMLWebScrapingStrategy(),
#         verbose=True
#     )

#     async with AsyncWebCrawler() as crawler:
#         results = await crawler.arun("https://www.dgsi.pt/jtre.nsf/8f8d2b7e72244bca80256879006d6594?CreateDocument", config=config)

#         print(f"Crawled {len(results)} pages in total")

#         # Access individual results
#         for result in results[:3]:  # Show first 3 results
#             print(f"URL: {result.url}")
#             print(f"Depth: {result.metadata.get('depth', 0)}")

# if __name__ == "__main__":
#     asyncio.run(main())


""" TODO: This website requires JS commands, to search keyworkds so it finds the correct data, NEED TO FIND A WAY TO PASS THE SEARCH KEWWORDS TO THE CRAWLER"""

import asyncio
import sys

# Fix para Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy

async def main():
    # URL com parâmetros de busca explícitos
    search_url = "https://www.dgsi.pt/jtre.nsf/134973db04f39bf2802579bf005f080b?SearchView&Query=VIOL%C3%8ANCIA%20DOM%C3%89STICA&SearchOrder=4"
    
    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=2,
            include_external=False,
            
        ),
        verbose=True
    )

    async with AsyncWebCrawler(verbose=True) as crawler:
        results = await crawler.arun(search_url, config=config)
        
        print(f"Crawled {len(results)} pages in total")
        
        for result in results[:5]:  # Mostrar primeiros 5 resultados
            print(f"URL: {result.url}")
            print(f"Depth: {result.metadata.get('depth', 0)}")
            if len(result.markdown) > 0:
                print(f"Content Preview: {result.markdown[:200]}...\n")
            else:
                print("No content found in this page.\n")

if __name__ == "__main__":
    asyncio.run(main())