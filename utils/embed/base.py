from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Generic, List, Optional, Tuple, TypeVar

import asyncio

from schemas.embed import EmbedResult

from utils.logger import _logger

S = TypeVar("S")
T = TypeVar("T")

logger = _logger(__name__)


class AbsEmbedder(ABC, Generic[S, T]):
    """
    Base class for text embedding
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        batch_size: int = 1,
        max_workers: int = 1,
        **kwargs
    ):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

        self.model_path = model_path

        if tokenizer_path:
            self.tokenizer_path = tokenizer_path

        temp = self._init_session_pool()
        self._sessions = temp[0]
        self._tokenizers = temp[1]

        self._session_index = 0

    @abstractmethod
    def _init_session_pool(self) -> Tuple[List[S], List[T]]:
        """
        Initialize sessions and tokenizers(if needed)
        """
        pass

    @property
    def session(self) -> Tuple[S, Optional[T]]:
        """
        Get session and tokenizer(if exists)
        """
        idx = self._session_index % self.max_workers
        session = self._sessions[idx]
        tokenizer = self._tokenizers[idx] if self._tokenizers else None

        self._session_index += 1

        return session, tokenizer

    @abstractmethod
    def inference(
        self,
        queries: List[str],
        **kwargs,
    ) -> List[EmbedResult]:
        pass

    def batch_inference(self, queries: List[str]) -> List[EmbedResult]:
        session, tokenizer = self.session

        def parts(_list, n):
            for idx in range(0, len(_list), n):
                yield _list[idx:idx + n]

        _queries = list(parts(queries, self.batch_size))

        results: List[EmbedResult] = []
        from tqdm import tqdm

        for qs in tqdm(_queries, desc="Batch inference"):
            results += self.inference(qs, session=session, tokenizer=tokenizer)

        return results

    async def inference_async(self, queries: str | List[str]):
        loop = asyncio.get_event_loop()
        queries = queries if isinstance(queries, list) else [queries]
        result = await loop.run_in_executor(self._pool, self.inference, queries)
        return result

    async def batch_inference_async(self,
                                    queries: List[str]) -> List[EmbedResult]:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._pool, self.batch_inference, queries
        )
        for e in result:
            if e.chunk:
                e.chunk = e.chunk.encode(errors="ignore").decode("utf-8")

        return result

    def release(self):
        self._pool.shutdown()
        for session in self._sessions:
            del session
        for tokenizer in self._tokenizers:
            del tokenizer
