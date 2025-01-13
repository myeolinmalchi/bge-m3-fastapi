from typing import List
from kss import Kss


def create_chunks(
    sentences: List[str], max_chunk_length: int, offset: int
) -> List[str]:

    if max_chunk_length < offset:
        raise Exception("max_chunk_length must be larger than offset.")

    chunks, current = [], ""
    for sentence in sentences:
        new_chunk = (
            current + " " + sentence.strip() if len(current) > 0 else sentence.strip()
        )
        if len(new_chunk) <= max_chunk_length:
            current = new_chunk
        else:
            chunks.append(new_chunk.strip())
            current = new_chunk[offset * -1]

    current = current.strip()
    chunks.append(current)

    return chunks


split_sentences = Kss("split_sentences")


def split_chunks(query: str, max_chunk_length: int = 1000, offset: int = 100):
    sentences = split_sentences(text=query, strip=True, backend="mecab")
    chunks = create_chunks(sentences, max_chunk_length, offset)
    return chunks
