from fastapi import FastAPI
from schemas.embed import EmbedRequest, EmbedResponse
from utils.onnxutils import ONNXRuntime, init_runtime
from utils.preprocess import split_chunks

from tqdm import tqdm


app = FastAPI()


@app.get("/")
def heath_check():
    return {"message": "everything is good"}


@app.post("/embed")
async def embed(req: EmbedRequest) -> EmbedResponse:
    runtime = ONNXRuntime.get_runtime()
    if isinstance(req.inputs, list):
        if not req.chunking:
            results = await runtime.abatch_inference(req.inputs)
        else:
            from itertools import chain, islice

            chunks = [split_chunks(i) for i in tqdm(req.inputs, desc="Chunking")]
            lens = [len(_chunks) for _chunks in chunks]
            flatten_chunks = list(chain(*chunks))

            temp = await runtime.abatch_inference(flatten_chunks)
            iterator = iter(temp)
            results = [list(islice(iterator, length)) for length in lens]

        return results

    if not req.chunking:
        result = await runtime.abatch_inference([req.inputs])
        return result[0]
    chunks = split_chunks(req.inputs)
    results = await runtime.abatch_inference(chunks)
    return results


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
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
        "-w",
        "--max-workers",
        dest="max_workers",
        action="store",
        default="1",
        help="(cuda) Number of inference sessions (default: 1)",
    )

    args = parser.parse_args()

    device = str(args.device)
    batch_size = int(args.batch_size)
    max_workers = int(args.max_workers)
    assert device in ("cpu", "cuda")
    init_runtime(device=device, batch_size=batch_size, max_workers=max_workers)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        timeout_keep_alive=600,
    )
