from typing import Any

from onnxruntime import InferenceSession
from utils.embed.base import AbsEmbedder


class ONNXEmbedder(AbsEmbedder[InferenceSession, Any]):
    pass
