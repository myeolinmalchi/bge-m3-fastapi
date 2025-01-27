from typing import Optional, List
from utils.embed.base import AbsEmbedder
from utils.embed.llama_cpp.session import LlamaCppSession
from utils.logger import _logger

logger = _logger(__name__)


class LlamaCppEmbedder(AbsEmbedder[LlamaCppSession, None]):

    def _init_session_pool(self):
        sessions = [
            LlamaCppSession(self.model_path) for _ in range(self.max_workers)
        ]

        return sessions, []

    def inference(
        self,
        queries: List[str],
        session: Optional[LlamaCppSession] = None,
        **kwargs
    ):
        if len(queries) > self.batch_size:
            raise Exception(
                f"최대 배치 크기를 초과한 입력({len(queries)} > {self.max_workers})입니다."
            )

        session = self.session[0] if session is None else session
        embeddings = session.embed_(queries)

        return embeddings
