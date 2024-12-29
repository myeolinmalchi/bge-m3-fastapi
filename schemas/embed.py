from typing import List, Dict, Optional
from pydantic import BaseModel


class EmbedRequest(BaseModel):
    inputs: List[str] | str
    truncate: bool = True
    chunking: bool = True


class EmbedResult(BaseModel):
    chunk: Optional[str] = None
    dense: List[float]
    sparse: Dict[int, float]


EmbedResponse = List[List[EmbedResult]] | List[EmbedResult] | EmbedResult
