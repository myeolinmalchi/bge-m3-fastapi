from fastapi import Depends, FastAPI
from schemas.embed import EmbedRequest, EmbedResponse
from utils.onnxutils import ONNXRuntime, init_runtime
from utils.preprocess import split_chunks

from tqdm import tqdm


app = FastAPI()


@app.get("/")
def heath_check():
    return {"message": "everything is good"}


@app.post("/embed")
def embed(req: EmbedRequest, runtime=Depends(ONNXRuntime.get_runtime)) -> EmbedResponse:
    if isinstance(req.inputs, list):
        if not req.chunking:
            results = runtime.parallel_execution(req.inputs)
        else:
            chunks = [split_chunks(i) for i in tqdm(req.inputs, desc="Chunking")]
            results = [
                runtime.parallel_execution(_chunks)
                for _chunks in tqdm(chunks, desc="ONNX")
            ]
        return results

    if not req.chunking:
        return runtime.parallel_execution(req.inputs)
    chunks = split_chunks(req.inputs)
    results = runtime.parallel_execution(chunks)
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
    )
    parser.add_argument(
        "-n", "--batch_size", dest="batch_size", action="store", default="2", type=int
    )
    args = parser.parse_args()

    device = str(args.device)
    batch_size = int(args.batch_size)
    assert device in ("cpu", "cuda")
    init_runtime(device=device, N=batch_size)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        timeout_keep_alive=600,
    )
