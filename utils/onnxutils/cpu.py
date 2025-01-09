from multiprocessing import cpu_count
from typing import Any, List, Optional
import onnxruntime as ort
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession

from schemas.embed import EmbedResult

from .base import ONNXRuntime
from transformers import AutoTokenizer


class ONNXCpuRuntime(ONNXRuntime):

        ]

        )
