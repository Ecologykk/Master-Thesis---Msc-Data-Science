
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


import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy

async def main():
    # Configure a 2-level deep crawl
    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=2, 
            include_external=False
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=True
    )

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun("https://www.dgsi.pt/jtre.nsf/8f8d2b7e72244bca80256879006d6594?CreateDocument", config=config)

        print(f"Crawled {len(results)} pages in total")

        # Access individual results
        for result in results[:3]:  # Show first 3 results
            print(f"URL: {result.url}")
            print(f"Depth: {result.metadata.get('depth', 0)}")

if __name__ == "__main__":
    asyncio.run(main())


""" TODO: This website requires JS commands, to search keyworkds so it finds the correct data, NEED TO FIND A WAY TO PASS THE SEARCH KEWWORDS TO THE CRAWLER"""

# import asyncio
# from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
# from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
# from crawl4ai.content_scraping_strategy import PlaywrightScrapingStrategy  # For JS/dynamic

# async def main():
#     # Use a proper search results URL
#     START_URL = "https://www.dgsi.pt/jtre.nsf/134973db04f39bf2802579bf005f080b?SearchView&Query=viol%C3%AAncia%20dom%C3%A9stica&SearchOrder=4"  # Example; test manually first

#     config = CrawlerRunConfig(
#         deep_crawl_strategy=BFSDeepCrawlStrategy(
#             max_depth=2,  # Level 0: results page; Level 1: individual acórdãos; Level 2: any sub-links
#             include_external=False,
#             max_links_per_page=20  # Limit to avoid too many
#         ),
#         scraping_strategy=PlaywrightScrapingStrategy(  # Handles potential JS
#             headless=True,  # Run without browser window
#             timeout=30000  # 30s timeout
#         ),
#         verbose=True
#     )

#     async with AsyncWebCrawler() as crawler:
#         results = await crawler.arun(START_URL, config=config)

#         print(f"Crawled {len(results)} pages in total")

#         # Save extracted content (e.g., to JSON for dataset)
#         import json
#         dataset = [{"url": r.url, "depth": r.metadata.get('depth', 0), "content": r.markdown} for r in results]
#         with open("acordaos_violencia_domestica.json", "w", encoding="utf-8") as f:
#             json.dump(dataset, f, ensure_ascii=False, indent=4)

#         # Preview first 3
#         for result in results[:3]:
#             print(f"URL: {result.url}")
#             print(f"Depth: {result.metadata.get('depth', 0)}")
#             print(f"Content Preview: {result.markdown[:200]}...\n")

# if __name__ == "__main__":
#     asyncio.run(main())