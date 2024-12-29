# bge-m3-fastapi

- Requirements: `poetry==^1.8`, `python==^3.10`
- Download the `model.onnx` and `model.onnx.data` files from [aapot/bge-m3-onnx](https://huggingface.co/aapot/bge-m3-onnx) and move them to the `models` directory.

## Usage

```bash
git clone https://github.com/myeolinmalchi/bge-m3-fastapi.git
cd bge-m3-fastapi

poetry install --no-root
uvicorn main:app --host=0.0.0.0 --port=8000 --timeout-keep-alive=6000
```
