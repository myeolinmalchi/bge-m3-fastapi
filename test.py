import asyncio
from utils.onnxutils import *

runtime = init_runtime(device="cuda", batch_size=8, max_workers=8)

inputs = ["테스트입니다." for _ in range(200)]

from time import time

st = time()
results1 = runtime.batch_inference(inputs)
print(f"Batch inference: {time() - st:.4f} sec")


async def ainference():
    futures = [runtime.ainference(input) for input in inputs]
    results = await asyncio.gather(*futures)
    return results


st = time()
results2 = asyncio.run(ainference())
print(f"Async inference: {time() - st:.4f} sec")
