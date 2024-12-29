from fastapi import FastAPI
from schemas.embed import EmbedRequest, EmbedResponse
from utils.onnx import run_onnx_pool
from utils.preprocess import split_chunks

from tqdm import tqdm


app = FastAPI()


@app.get("/")
def heath_check():
    return {"message": "everything is good"}


@app.post("/embed")
def embed(req: EmbedRequest) -> EmbedResponse:
    if isinstance(req.inputs, list):
        if not req.chunking:
            results = run_onnx_pool(req.inputs)
        else:
            chunks = [split_chunks(i) for i in tqdm(req.inputs, desc="Chunking")]
            results = [run_onnx_pool(_chunks) for _chunks in tqdm(chunks, desc="ONNX")]
        return results

    if not req.chunking:
        return run_onnx_pool(req.inputs)
    chunks = split_chunks(req.inputs)
    results = run_onnx_pool(chunks)

    return results
