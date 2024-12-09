import re
import asyncio

def split_text_into_chunks(text, chunk_size):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        if sum(len(s) for s in current_chunk) + len(sentence) + 1 > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
        else:
            current_chunk.append(sentence)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

async def process_text_content(texts, chunk_size):
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, split_text_into_chunks, text, chunk_size) for text in texts]
    return await asyncio.gather(*tasks)
