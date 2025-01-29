"""텍스트 전처리 유틸 함수 모음"""

from typing import List
from kss import Kss
from kss._utils.logger import logger

split_sentences = Kss("split_sentences")
normalize = Kss("normalize")


def create_chunks(sentences: List[str], max_chunk_length: int,
                  offset: int) -> List[str]:

    if max_chunk_length < offset:
        raise Exception("max_chunk_length must be larger than offset.")

    chunks, current = [], ""
    for sentence in sentences:
        new_chunk = (
            current + " " +
            sentence.strip() if len(current) > 0 else sentence.strip()
        )
        if len(new_chunk) <= max_chunk_length:
            current = new_chunk
        else:
            chunks.append(new_chunk.strip())
            current = new_chunk[offset * -1]

    current = current.strip()
    chunks.append(current)

    return chunks


def preprocess_single(text: str):
    import re
    text = re.sub(r"\\+", "", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\t", "", text)
    text = re.sub(r"\r", "", text)
    exclude_base64 = re.compile(r"data:image/[a-zA-Z]+;base64,[^\"']+")
    text = re.sub(exclude_base64, "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess(text: str | List[str], **kwargs) -> str | List[str]:
    """전처리 함수

    다른 로직 추가될 수 있어서 따로 분리함.
    """
    inputs = text if isinstance(text, list) else [text]
    texts = []
    for i in inputs:
        temp = normalize(
            i,
            allow_doubled_spaces=False,
            allow_html_tags=False,
            allow_html_escape=False,
            allow_halfwidth_hangul=False,
            allow_hangul_jamo=False,
            allow_invisible_chars=False,
            reduce_char_repeats_over=2,
            reduce_emoticon_repeats_over=2
        )
        texts.append(temp)
    """텍스트 전처리"""

    cleaned = [preprocess_single(t) for t in texts]
    return cleaned[0] if isinstance(text, str) else cleaned


def split_chunks(query: str, max_chunk_length: int = 1000, offset: int = 100):
    sentences = split_sentences(text=query, strip=True, backend="mecab")
    chunks = create_chunks(sentences, max_chunk_length, offset)
    return chunks
