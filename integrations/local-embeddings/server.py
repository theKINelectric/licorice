#!/usr/bin/env python3
"""Private CPU embedding sidecar. Loads only hash-pinned local model assets."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import os
from pathlib import Path
import socket
import threading
import time

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_BATCH = 40
MAX_ITEM_BYTES = 4000
MAX_INPUT_BYTES = 160000
MAX_BODY_BYTES = 200000
DIMENSIONS = 384
MAX_SEQUENCE = 256
HERE = Path(__file__).resolve().parent


class ServiceError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code = status, code


def verify_assets(directory, manifest):
    """Read local files only; no hub resolver or fallback downloader exists."""
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid manifest path")
        path = directory / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError("model asset missing or size mismatch")
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        if hasher.hexdigest() != entry["sha256"]:
            raise ValueError("model asset checksum mismatch")


def validate_input(body):
    if not isinstance(body, dict) or set(body) - {"model", "input", "encoding_format", "kind"}:
        raise ServiceError(400, "invalid_request", "Expected only model, input and optional encoding_format or kind.")
    if body.get("model") != MODEL or body.get("encoding_format", "float") != "float" or body.get("kind", "document") not in ("query", "document"):
        raise ServiceError(400, "invalid_model", "Use the configured model and floating-point encoding.")
    texts = body.get("input")
    if not isinstance(texts, list) or not 1 <= len(texts) <= MAX_BATCH:
        raise ServiceError(400, "invalid_batch", "Input must be an array of 1 to 40 strings.")
    total = 0
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            raise ServiceError(400, "invalid_input", "Each input must be nonempty text.")
        try:
            length = len(text.encode("utf-8"))
        except UnicodeEncodeError:
            raise ServiceError(400, "invalid_input", "Input must be valid UTF-8 text.") from None
        if length > MAX_ITEM_BYTES:
            raise ServiceError(413, "input_limit", "Each input is limited to 4000 UTF-8 bytes.")
        total += length
    if total > MAX_INPUT_BYTES:
        raise ServiceError(413, "input_limit", "Combined input is limited to 160000 UTF-8 bytes.")
    return texts


def mean_pool_normalize(hidden, mask, np):
    if hidden.ndim != 3 or hidden.shape[:2] != mask.shape or hidden.shape[2] != DIMENSIONS:
        raise ServiceError(503, "invalid_output", "The embedding runtime returned an invalid tensor.")
    expanded = mask[:, :, None].astype(np.float32)
    pooled = (hidden.astype(np.float32) * expanded).sum(axis=1) / np.maximum(expanded.sum(axis=1), 1e-9)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    if not np.isfinite(pooled).all() or not np.isfinite(norms).all() or (norms <= 1e-12).any():
        raise ServiceError(503, "invalid_output", "The embedding runtime returned an unusable vector.")
    result = pooled / norms
    if not np.isfinite(result).all():
        raise ServiceError(503, "invalid_output", "The embedding runtime returned an unusable vector.")
    return result


class Embedder:
    def __init__(self, tokenizer, session, np, run_options_factory, manifest, *, timeout=8.0, runtime_version="fixture"):
        self.tokenizer, self.session, self.np = tokenizer, session, np
        self.run_options_factory = run_options_factory
        self.manifest = manifest
        self.timeout = timeout
        self.runtime_version = runtime_version
        self.running = threading.BoundedSemaphore(1)
        inputs = {value.name for value in session.get_inputs()}
        if not {"input_ids", "attention_mask"} <= inputs or inputs - {"input_ids", "attention_mask", "token_type_ids"}:
            raise ValueError("unexpected ONNX inputs")
        self.input_names = inputs
        if "last_hidden_state" not in {value.name for value in session.get_outputs()}:
            raise ValueError("expected token embedding output")

    def health(self):
        return {"status": "ok", "model": MODEL, "revision": self.manifest["revision"], "dimensions": DIMENSIONS,
                "pooling": "attention-mask mean then L2 normalization", "max_sequence_length": MAX_SEQUENCE,
                "query_prefix": "", "document_prefix": "", "provider": "CPUExecutionProvider",
                "runtime_version": self.runtime_version, "assets_verified": True, "runtime_downloads": False,
                "limits": {"batch": MAX_BATCH, "item_utf8_bytes": MAX_ITEM_BYTES, "input_utf8_bytes": MAX_INPUT_BYTES, "http_body_bytes": MAX_BODY_BYTES, "inference_timeout_seconds": self.timeout}}

    def embed(self, body):
        texts = validate_input(body)
        if not self.running.acquire(blocking=False):
            raise ServiceError(503, "busy", "The local embedding runtime is busy; no request was queued.")
        timer = None
        try:
            started = time.monotonic()
            options = self.run_options_factory()
            encoded = self.tokenizer.encode_batch(texts)
            ids = self.np.asarray([row.ids for row in encoded], dtype=self.np.int64)
            mask = self.np.asarray([row.attention_mask for row in encoded], dtype=self.np.int64)
            if ids.ndim != 2 or ids.shape != mask.shape or ids.shape[0] != len(texts) or not 1 <= ids.shape[1] <= MAX_SEQUENCE:
                raise ServiceError(503, "invalid_tokenization", "Local tokenization returned an invalid shape.")
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feeds["token_type_ids"] = self.np.asarray([row.type_ids for row in encoded], dtype=self.np.int64)
            remaining = self.timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise ServiceError(504, "timeout", "The local embedding deadline expired.")
            timer = threading.Timer(remaining, lambda: setattr(options, "terminate", True))
            timer.daemon = True
            timer.start()
            try:
                hidden = self.session.run(["last_hidden_state"], feeds, options)[0]
            except Exception:
                if options.terminate or time.monotonic() - started >= self.timeout:
                    raise ServiceError(504, "timeout", "The local embedding deadline expired.") from None
                raise ServiceError(503, "runtime_failure", "The local embedding operation failed.") from None
            if time.monotonic() - started >= self.timeout:
                raise ServiceError(504, "timeout", "The local embedding deadline expired.")
            vectors = mean_pool_normalize(hidden, mask, self.np)
            return {"object": "list", "model": MODEL, "dimensions": DIMENSIONS,
                    "data": [{"object": "embedding", "index": index, "embedding": vector.tolist()} for index, vector in enumerate(vectors)],
                    "tokenization": {"max_sequence_length": MAX_SEQUENCE, "truncated_input_indices": [index for index, row in enumerate(encoded) if row.overflowing], "encoded_tokens": [int(row.attention_mask.count(1)) for row in encoded]},
                    "usage": {"input_tokens": int(mask.sum())}}
        finally:
            if timer is not None:
                timer.cancel()
            self.running.release()


def load_model(directory, *, timeout=8.0, threads=2):
    manifest = json.loads((HERE / "model-manifest.json").read_text())
    verify_assets(directory, manifest)
    if json.loads((directory / "sentence_bert_config.json").read_text())["max_seq_length"] != MAX_SEQUENCE:
        raise ValueError("unexpected model sequence limit")
    pooling = json.loads((directory / "1_Pooling/config.json").read_text())
    if not pooling["pooling_mode_mean_tokens"] or pooling["pooling_mode_cls_token"]:
        raise ValueError("unexpected model pooling")
    # These libraries operate on explicit local files, never from_pretrained.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer
    ort.disable_telemetry_events()
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=MAX_SEQUENCE)
    tokenizer.enable_padding(direction="right", pad_id=0, pad_type_id=0, pad_token="[PAD]")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.log_severity_level = 3
    session = ort.InferenceSession(str(directory / "onnx/model.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    return Embedder(tokenizer, session, np, ort.RunOptions, manifest, timeout=timeout, runtime_version=ort.__version__)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "LicoriceLocalEmbedding"
    sys_version = ""

    def log_message(self, *_):
        pass  # Never log paths, payloads, embeddings or request headers.

    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def reply(self, status, body):
        self.close_connection = True
        payload = json.dumps(body, separators=(",", ":"), allow_nan=False).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
        except (OSError, ConnectionError):
            pass

    def do_GET(self):
        if self.path == "/health":
            self.reply(200, self.server.embedder.health())
        else:
            self.reply(404, {"error": {"code": "not_found", "message": "No such local route."}})

    def do_POST(self):
        try:
            if self.path not in ("/v1/embeddings", "/embeddings"):
                raise ServiceError(404, "not_found", "No such local route.")
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or self.headers.get("Transfer-Encoding") or not lengths[0].isdigit():
                raise ServiceError(400, "invalid_length", "Send one bounded Content-Length, without transfer encoding.")
            length = int(lengths[0])
            if not 1 <= length <= MAX_BODY_BYTES:
                raise ServiceError(413, "body_limit", "The local request body is too large.")
            if self.headers.get_content_type() != "application/json":
                raise ServiceError(415, "content_type", "Send application/json.")
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ServiceError(400, "invalid_body", "The request body was incomplete.")
            body = json.loads(raw.decode("utf-8"))
            self.reply(200, self.server.embedder.embed(body))
        except ServiceError as error:
            self.reply(error.status, {"error": {"code": error.code, "message": str(error)}})
        except (ValueError, UnicodeDecodeError):
            self.reply(400, {"error": {"code": "invalid_json", "message": "Send valid JSON with UTF-8 text."}})
        except (socket.timeout, TimeoutError):
            self.reply(408, {"error": {"code": "read_timeout", "message": "The local request read deadline expired."}})
        except Exception:
            self.reply(503, {"error": {"code": "runtime_failure", "message": "The local embedding operation failed."}})


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 4

    def __init__(self, address, embedder):
        self.embedder = embedder
        self.connections = threading.BoundedSemaphore(4)
        super().__init__(address, Handler)

    def process_request(self, request, client_address):
        if not self.connections.acquire(blocking=False):
            try:
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()

    def handle_error(self, request, client_address):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--allow-private-bind", action="store_true", help="Permit a private-network bind; requires operator network isolation")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    address = ipaddress.ip_address(args.host)
    if address.version != 4 or not (address.is_loopback or (args.allow_private_bind and (address.is_private or address.is_unspecified))):
        parser.error("bind only to IPv4 loopback or an explicitly isolated private network")
    if not args.model_dir.is_absolute() or not args.model_dir.is_dir() or not 1 <= args.port <= 65535 or not 0.1 <= args.timeout <= 30 or not 1 <= args.threads <= 4:
        parser.error("invalid local model directory or resource limits")
    try:
        embedder = load_model(args.model_dir, timeout=args.timeout, threads=args.threads)
    except Exception:
        parser.exit(1, "Local embedding startup failed: pinned assets or CPU runtime could not be verified.\n")
    with Server((args.host, args.port), embedder) as server:
        print(json.dumps(embedder.health(), sort_keys=True), flush=True)
        try:
            server.serve_forever(poll_interval=0.1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
