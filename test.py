from utils.onnxutils import *

runtime1 = init_runtime(device="cpu", N=2)
runtime2 = ONNXPoolExecutor("BAAI/bge-m3", "models/model.onnx")
runtime3 = ONNXCudaRuntime("BAAI/bge-m3", "models/model.onnx")

print(runtime1)
print(runtime2)
print(runtime3)
