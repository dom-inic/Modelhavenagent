import aiohttp
import asyncio
from bs4 import BeautifulSoup

async def fetch(session, url, headers, params=None):
    async with session.get(url, params=params, headers=headers, timeout=30) as response:
        return await response.text()

async def fetch_page(session, params, page_num, results, total_results_to_fetch, headers):
    params["start"] = (page_num - 1) * params["num"]
    html = await fetch(session, "https://www.google.com/search", headers, params)
    soup = BeautifulSoup(html, 'html.parser')

    for result in soup.select(".tF2Cxc"):
        if len(results) >= total_results_to_fetch:
            break
        title = result.select_one(".DKV0Md").text
        links = result.select_one(".yuRUbf a")["href"]

        results.append({"title": title, "links": links})

async def fetch_content(session, url, headers):
    async with session.get(url, headers=headers, timeout=30) as response:
        return await response.text()

async def fetch_all_content(urls, headers):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_content(session, url, headers) for url in urls]
        return await asyncio.gather(*tasks)
