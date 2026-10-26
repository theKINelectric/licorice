ARG PYTHON_IMAGE
FROM ${PYTHON_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false ORT_DISABLE_TELEMETRY=1
WORKDIR /app
COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir --require-hashes --only-binary=:all: -r /app/requirements.lock && useradd --system --uid 10001 --no-create-home embeddings
COPY server.py model-manifest.json /app/
COPY notices/ /app/notices/
COPY runtime-license-inventory.json license-obligations.md /app/notices/
USER 10001:10001
# Mount the separately acquired, verified model read-only at /models/minilm.
# No model is downloaded at build or runtime. Do not publish this private port.
ENTRYPOINT ["python", "/app/server.py", "--model-dir", "/models/minilm", "--host", "0.0.0.0", "--allow-private-bind", "--port", "8091"]
