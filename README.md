# bge-m3-fastapi

- Requirements: `poetry==^1.8`, `python==^3.10`
- Download the `model.onnx` and `model.onnx.data` files from [aapot/bge-m3-onnx](https://huggingface.co/aapot/bge-m3-onnx) and move them to the `models` directory.

## Usage

1. Clone this repository
    ```bash
    git clone https://github.com/myeolinmalchi/bge-m3-fastapi.git
    cd bge-m3-fastapi
    ```

2. Install dependencies
    ```bash
    poetry shell
    poetry install --no-root
    ```

3. Run `main.py`
    ```bash
    poetry run python3.10 main.py --device <device type> --batch_size <batch size>
        --device: device type for onnxruntime (cpu/cuda, default: cpu)
        --batch_size: concurrency limit for onnxruntime (default: 2)
    ```
