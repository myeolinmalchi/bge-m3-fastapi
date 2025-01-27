from fastapi import FastAPI, HTTPException
from schemas.embed import EmbedRequest, EmbedResponse
from utils.embed.base import AbsEmbedder
from utils.preprocess import split_chunks
from utils.embed import init_runtime

from tqdm import tqdm

app = FastAPI()


@app.get("/")
def heath_check():
    return {"message": "everything is good"}


@app.post("/embed")
async def embed(req: EmbedRequest) -> EmbedResponse:
    try:
        runtime = AbsEmbedder.get_runtime()
        if isinstance(req.inputs, list):
            if not req.chunking:
                results = await runtime.batch_inference_async(req.inputs)
            else:
                from itertools import chain, islice

                chunks = [
                    split_chunks(i) for i in tqdm(req.inputs, desc="Chunking")
                ]
                lens = [len(_chunks) for _chunks in chunks]
                flatten_chunks = list(chain(*chunks))

                temp = await runtime.batch_inference_async(flatten_chunks)
                iterator = iter(temp)
                results = [list(islice(iterator, length)) for length in lens]

            return results

        if not req.chunking:
            result = await runtime.batch_inference_async([req.inputs])
            return result[0]
        chunks = split_chunks(req.inputs)
        results = await runtime.batch_inference_async(chunks)
        return results
    except Exception:
        raise HTTPException(status_code=400, detail="Error has been occurred")


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--backend",
        dest="backend",
        action="store",
        default="onnx",
        choices=["onnx", "llama_cpp"],
        help=
        "Witch backend to use for inference (onnx/llama_cpp, default: onnx)",
    )
    parser.add_argument(
        "-d",
        "--device",
        dest="device",
        action="store",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Witch device to use for inference (cpu/cuda, default: cpu)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        action="store",
        default="1",
        help="Max batch size for inference (default: 1)",
    )
    parser.add_argument(
        "-m",
        "--model-path",
        dest="model_path",
        action="store",
        default="",
        help="Path to model file",
    )
    parser.add_argument(
        "-w",
        "--max-workers",
        dest="max_workers",
        action="store",
        default="1",
        help="(cuda) Number of inference sessions (default: 1)",
    )

    args = parser.parse_args()

    device = str(args.device)
    backend = str(args.backend)
    batch_size = int(args.batch_size)
    max_workers = int(args.max_workers)
    model_path = str(args.model_path)
    assert device in ("cpu", "cuda")
    assert backend in ("onnx", "llama_cpp")

    init_runtime(
        device=device,
        batch_size=batch_size,
        max_workers=max_workers,
        backend=backend,
        model_path=model_path
    )

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        timeout_keep_alive=600,
    )
