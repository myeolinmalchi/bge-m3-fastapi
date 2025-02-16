from fastapi import Depends, FastAPI, HTTPException
from schemas.embed import EmbedRequest, EmbedResponse
from utils.embed.base import AbsEmbedder
from utils.preprocess.text import md
from utils.text import preprocess, split_chunks
from utils.embed import init_runtime
import cli

app = FastAPI()


@app.get("/")
def heath_check():
    return {"message": "everything is good"}


@app.post("/embed")
async def embed(
    req: EmbedRequest, runtime: AbsEmbedder = Depends(init_runtime)
) -> EmbedResponse:
    from itertools import chain, islice

    try:
        match req:
            case EmbedRequest(inputs=list(inputs), html=True):
                markdowns = [md(input) for input in inputs]
                results = await runtime.batch_inference_async(markdowns)
                response = [[result] for result in results]

            case EmbedRequest(inputs=list(inputs), chunking=True, html=False):
                chunks = [preprocess(chunk) for chunk in inputs]
                lens = [len(_chunks) for _chunks in chunks]
                flatten_chunks = list(chain(*chunks))

                iterator = iter(
                    await runtime.batch_inference_async(flatten_chunks)
                )

                response = [list(islice(iterator, length)) for length in lens]

            case EmbedRequest(inputs=str(input), chunking=False):
                cleaned = preprocess(input)

                assert isinstance(cleaned, str)

                result = await runtime.batch_inference_async([cleaned])
                response = result[0]

            case EmbedRequest(inputs=str(input), chunking=True):
                chunks = split_chunks(input)
                cleaned = preprocess(chunks)

                assert isinstance(cleaned, list)

                response = await runtime.batch_inference_async(cleaned)

            case _:
                response = []

        return response

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
