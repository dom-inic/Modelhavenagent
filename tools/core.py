from concurrent.futures import ThreadPoolExecutor
from langchain_core.documents import Document
import aiohttp
import asyncio
from .fetcher import fetch_page
from .processor import process_text_content
from .common import get_all_text_from_url
from .vector_store import vector_store

chunk_size = 1024  # size of each text chunk
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
}

async def fetch_and_process_data(search_query,chunk_size=chunk_size,headers=headers,n_result_per_page=3,total_results_to_fetch=3):
    params = {
        "q": search_query,  # query example
        "hl": "en",         # language
        "gl": "uk",         # country of the search, UK -> United Kingdom
        "start": 0,         # number page by default up to 0
        "num": n_result_per_page    # parameter defines the maximum number of results to return per page.
    }

    async with aiohttp.ClientSession() as session:
        page_num = 0
        results = []
        while len(results) < total_results_to_fetch:
            page_num += 1
            await fetch_page(session, params, page_num, results,total_results_to_fetch,headers)

        urls = [result['links'] for result in results]

        with ThreadPoolExecutor(max_workers=10) as executor:
            loop = asyncio.get_event_loop()
            texts = await asyncio.gather(
                *[loop.run_in_executor(executor, get_all_text_from_url, url ,headers) for url in urls]
            )

        chunks_list = await process_text_content(texts, chunk_size)

        documents = []
        for i, result in enumerate(results):
            for j, chunk in enumerate(chunks_list[i]):
                documents.append(Document(page_content=chunk , metadata={'source': result['links'] ,
                                                                         'title': result['title']}))
        vector_store.add_documents(documents=documents)

    return documents