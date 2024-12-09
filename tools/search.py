import asyncio
from core import fetch_and_process_data
from vector_store import vector_store

def web_search(search_query: str):
    async def run_search():
        await fetch_and_process_data(search_query)
        results_ = vector_store.as_retriever().invoke(search_query)
        result_text = " ".join([results_[i].page_content for i in range(len(results_))])
        return result_text

    return asyncio.run(run_search())

result_text = web_search("LLama3")
print(result_text)
