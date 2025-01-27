FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

WORKDIR /app
RUN \
      apt-get update -y && \
      apt install software-properties-common -y && \
      add-apt-repository ppa:deadsnakes/ppa && \
      apt install python3.10 curl -y 
RUN curl -sSL https://install.python-poetry.org | python3 - 

ENV PATH="/root/.local/bin:$PATH"

COPY . /app
COPY pyproject.toml poetry.lock /app/

RUN \
      poetry config virtualenvs.create false && \
      poetry install --no-root --without dev

EXPOSE 8000

ENTRYPOINT ["poetry", "run", "python3.10", "main.py"]
CMD ["--device", "cpu", "--batch-size", "1", "--max-workers", "2", "--backend", "llama_cpp", "--model-path", "models/bge-m3-q8_0.gguf"]
