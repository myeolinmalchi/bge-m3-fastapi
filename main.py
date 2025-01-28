from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException
from schemas.embed import EmbedRequest, EmbedResponse
from utils.text import preprocess, split_chunks
from utils.embed import init_runtime

from tqdm import tqdm

runtime = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    args = {
        "device": "cpu",
        "backend": "llama_cpp",
        "batch_size": 4,
        "max_workers": 2,
        "model_path": "models/bge-m3-f16.gguf",
    }

    runtime = init_runtime(**args)
    yield
    runtime.release()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def heath_check():
    return {"message": "everything is good"}


@app.post("/embed")
async def embed(
    req: EmbedRequest, runtime=Depends(init_runtime)
) -> EmbedResponse:
    try:
        cleaned = preprocess(req.inputs)
        if isinstance(cleaned, list):
            if not req.chunking:
                results = await runtime.batch_inference_async(cleaned)
            else:
                from itertools import chain, islice

                chunks = [
                    split_chunks(i) for i in tqdm(cleaned, desc="Chunking")
                ]
                lens = [len(_chunks) for _chunks in chunks]
                flatten_chunks = list(chain(*chunks))

                temp = await runtime.batch_inference_async(flatten_chunks)
                iterator = iter(temp)
                results = [list(islice(iterator, length)) for length in lens]

            return results

        if not req.chunking:
            result = await runtime.batch_inference_async([cleaned])
            return result[0]
        chunks = split_chunks(cleaned)
        results = await runtime.batch_inference_async(chunks)
        return results
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error has been occurred {e}"
        )
