FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

WORKDIR /app
RUN \
      apt-get update -y && \
      apt install software-properties-common -y && \
      add-apt-repository ppa:deadsnakes/ppa && \
      apt install python3.10 curl -y && \
      apt install build-essential -y

RUN \
      apt install musl-dev -y && \
      ln -s /usr/lib/x86_64-linux-musl/libc.so /lib/libc.musl-x86_64.so.1


RUN curl -sSL https://install.python-poetry.org | python3 - 

ENV PATH="/root/.local/bin:$PATH"

COPY . /app
COPY pyproject.toml /app/

ENV CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"

ENV CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"

RUN \
      poetry config virtualenvs.create false && \
      poetry install --no-root --without dev

EXPOSE 8000

ENTRYPOINT ["poetry", "run", "python3.10", "main.py"]
CMD ["--device", "cpu", "--batch-size", "1", "--backend", "llama_cpp", "--sessions", "1", "--model-path", "models/bge-m3-q8_0.gguf"]
