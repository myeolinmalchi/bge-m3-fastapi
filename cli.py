import argparse
from typing import Literal, NotRequired, TypedDict


class Args(TypedDict):
    device: Literal["cpu", "cuda"]
    backend: Literal["onnx", "llama_cpp"]
    model_path: str
    tokenizer_path: NotRequired[str]

    batch_size: int
    sessions: int


def init_server_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--device",
        dest="device",
        action="store",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for inference. (cpu/cuda, default: cpu)"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        action="store",
        default="1",
        type=int,
        help="Batch size for inference. (default: 1)"
    )
    parser.add_argument(
        "-f",
        "--backend",
        dest="backend",
        action="store",
        default="onnx",
        choices=["onnx", "llama_cpp"],
        help="Inference backend to use. (onnx/llama_cpp, default: onnx)"
    )
    parser.add_argument(
        "-s",
        "--sessions",
        dest="sessions",
        action="store",
        default="1",
        type=int,
        help="Number of session instances for parallel processing. (default: 1)"
    )
    parser.add_argument(
        "-m",
        "--model-path",
        dest="model_path",
        action="store",
        type=str,
        help="Path to the model file. Required for model inference."
    )

    parser.add_argument(
        "-t",
        "--tokenizer-path",
        dest="tokenizer_path",
        action="store",
        type=str,
        help="Path to the tokenizer file."
    )

    args = parser.parse_args()

    return Args(**args.__dict__)
