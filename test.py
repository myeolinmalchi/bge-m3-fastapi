from utils.onnx import ONNXPoolExecutor, run_onnx_pool
from multiprocessing import cpu_count

from utils.preprocess import split_chunks

CPUs = cpu_count()
pool = ONNXPoolExecutor(N=2)

if __name__ == "__main__":
    pool.init_task()
    queries = [
        "테스트입니다. 테스트입니다. 테스트입니다. 테스트입니다. 테스트입니다. 테스트입니다. 테스트입니다. 테스트입니다. 테스트입니다. 테스트입니다. 테스트입니다. 테스트입니다. 테스트입니다. 테스트입니다. 테스트입니다."
        for _ in range(30)
    ]

    chunks = [split_chunks(query) for query in queries]
    results = [run_onnx_pool(_chunks) for _chunks in chunks]
    pool.release_task()
    from pprint import pprint

    pprint(results)

    print(len(results))
