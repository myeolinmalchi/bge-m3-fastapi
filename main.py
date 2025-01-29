from fastapi import Depends, FastAPI, HTTPException
from schemas.embed import EmbedRequest, EmbedResponse
from utils.text import preprocess, split_chunks
from utils.embed import init_runtime
import cli

from tqdm import tqdm

runtime = None

app = FastAPI()


@app.get("/")
def heath_check():
    return {"message": "everything is good"}


@app.post("/embed")
async def embed(
    req: EmbedRequest, runtime=Depends(init_runtime)
) -> EmbedResponse:
    try:
        inputs = req.inputs
        if isinstance(inputs, list):
            if not req.chunking:
                inputs = preprocess(inputs)
                results = await runtime.batch_inference_async(inputs)
            else:
                from itertools import chain, islice

                chunks = [
                    split_chunks(i) for i in tqdm(inputs, desc="Chunking")
                ]

                chunks = [preprocess(chunk) for chunk in chunks]
                lens = [len(_chunks) for _chunks in chunks]
                flatten_chunks = list(chain(*chunks))

                temp = await runtime.batch_inference_async(flatten_chunks)
                iterator = iter(temp)
                results = [list(islice(iterator, length)) for length in lens]

            return results

        if not req.chunking:
            cleaned = preprocess(inputs)
            result = await runtime.batch_inference_async([cleaned])
            return result[0]
        chunks = split_chunks(inputs)
        cleaned = preprocess(chunks)
        results = await runtime.batch_inference_async(cleaned)
        return results
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error has been occurred {e}"
        )


if __name__ == "__main__":
    import uvicorn

    args = cli.init_server_args()

    init_runtime(**args)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        timeout_keep_alive=600,
        workers=1
    )
