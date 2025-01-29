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


if __name__ == "__main__":
    import uvicorn

    args = cli.init_server_args()

    init_runtime(
        model_path=args['model_path'],
        batch_size=args['batch_size'],
        max_workers=args['sessions'],
        backend=args['backend'],
        device=args['device']
    )

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        timeout_keep_alive=600,
        workers=args['workers']
    )
