import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import argparse
import os
import multiprocessing as mp
async def search_on_baike(query, output_directory='.', filename=None):
    
    if filename is None:
        filename = f'{query}.txt'
    filepath = os.path.join(output_directory, filename)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.goto('https://baike.baidu.com/')

        await page.wait_for_selector('#root > div > div.index-module_pageHeader__jSG5w > div.lemmaSearchBarWrapper.undefined > div > div > div > div > input', timeout=5000)
        await page.fill('#root > div > div.index-module_pageHeader__jSG5w > div.lemmaSearchBarWrapper.undefined > div > div > div > div > input', query)
        await page.click('#root > div > div.index-module_pageHeader__jSG5w > div.lemmaSearchBarWrapper.undefined > div > div > div > button.lemmaBtn')

        await page.wait_for_timeout(5000)
        html_content = await page.content()
        soup = BeautifulSoup(html_content, 'html.parser')
        content_div = soup.find('div', {'class': 'J-lemma-content'})
        content_text = content_div.get_text().strip()
        print(content_text)

        # Save content_text to the specified file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content_text)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Search on Baike and save the content.")
#     parser.add_argument("--query", help="The keyword to search for.")
#     parser.add_argument("--output", default='.', help="Output directory (default: current directory).")
#     parser.add_argument("--filename", help="Optional, name of the output file (default: <query>.txt)")
#
#     args = parser.parse_args()
#
#     asyncio.run(search_on_baike(args.query, args.output, args.filename))