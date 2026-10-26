#!/usr/bin/env python3
"""Synthetic tokenizer/ORT controls; not a model-quality qualification."""
import hashlib
import http.client
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location("local_embeddings", Path(__file__).with_name("server.py"))
SIDE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SIDE)


class FakeTokenizer:
    def encode_batch(self, texts):
        return [SimpleNamespace(ids=[1, 2, 0], attention_mask=[1, 1, 0], type_ids=[0, 0, 0], overflowing=(["extra"] if len(text) > 100 else [])) for text in texts]


class FakeSession:
    def get_inputs(self):
        return [SimpleNamespace(name=name) for name in ["input_ids", "attention_mask", "token_type_ids"]]

    def get_outputs(self):
        return [SimpleNamespace(name="last_hidden_state")]

    def run(self, names, feeds, options):
        hidden = np.zeros((len(feeds["input_ids"]), 3, 384), dtype=np.float32)
        hidden[:, 0, :2] = [3, 0]
        hidden[:, 1, :2] = [0, 4]
        hidden[:, 2, :2] = [1000, 1000]
        return [hidden]


def make_engine(*, timeout=0.3, session=None):
    return SIDE.Embedder(FakeTokenizer(), session or FakeSession(), np, lambda: SimpleNamespace(terminate=False), {"revision": "synthetic-fixture"}, timeout=timeout)


def body(texts=None):
    return {"model": SIDE.MODEL, "input": texts or ["A synthetic sentence."], "encoding_format": "float"}


class PureControls(unittest.TestCase):
    def test_masked_mean_pool_ignores_padding_and_normalizes(self):
        result = make_engine().embed(body())
        vector = np.array(result["data"][0]["embedding"])
        np.testing.assert_allclose(vector[:2], [0.6, 0.8], atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(vector), 1.0, atol=1e-6)
        self.assertEqual(len(vector), 384)
        self.assertEqual(result["data"][0]["index"], 0)

    def test_nonfinite_or_wrong_shape_rejected(self):
        for value in [np.zeros((1, 3, 383)), np.full((1, 3, 384), np.nan), np.zeros((1, 3, 384))]:
            with self.assertRaises(SIDE.ServiceError):
                SIDE.mean_pool_normalize(value, np.ones((1, 3)), np)

    def test_strict_model_format_input_and_utf8_bounds(self):
        cases = [{"model": "wrong", "input": ["text"]}, {"model": SIDE.MODEL, "input": "text"}, body(["text"] * 41), body(["🙂" * 1001]), body([" "]), body([12]), {**body(), "encoding_format": "base64"}, {**body(), "api_key": "do-not-forward"}]
        for value in cases:
            with self.subTest(value=str(value)[:60]), self.assertRaises(SIDE.ServiceError):
                SIDE.validate_input(value)

    def test_batch_order_and_truncation_disclosed(self):
        result = make_engine().embed(body(["first", "long " * 30, "third"]))
        self.assertEqual([item["index"] for item in result["data"]], [0, 1, 2])
        self.assertEqual(result["tokenization"]["truncated_input_indices"], [1])
        self.assertEqual(result["usage"]["input_tokens"], 6)

    def test_busy_rejected_without_queue(self):
        engine = make_engine()
        engine.running.acquire()
        try:
            with self.assertRaises(SIDE.ServiceError) as caught:
                engine.embed(body())
            self.assertEqual(caught.exception.code, "busy")
        finally:
            engine.running.release()

    def test_run_options_factory_failure_releases_inference_slot(self):
        engine = make_engine()
        def fail_factory():
            raise RuntimeError("synthetic options allocation failure")
        engine.run_options_factory = fail_factory
        with self.assertRaises(RuntimeError):
            engine.embed(body())
        self.assertTrue(engine.running.acquire(blocking=False))
        engine.running.release()

    def test_run_deadline_sets_actual_termination_and_releases_slot(self):
        class SlowSession(FakeSession):
            def run(self, names, feeds, options):
                self.options = options
                deadline = time.monotonic() + 1
                while not options.terminate and time.monotonic() < deadline:
                    time.sleep(0.002)
                raise RuntimeError("private raw runtime details")
        session = SlowSession()
        engine = make_engine(timeout=0.02, session=session)
        with self.assertRaises(SIDE.ServiceError) as caught:
            engine.embed(body())
        self.assertEqual(caught.exception.status, 504)
        self.assertNotIn("private raw", str(caught.exception))
        self.assertTrue(session.options.terminate)
        self.assertTrue(engine.running.acquire(blocking=False))
        engine.running.release()

    def test_asset_mismatch_and_missing_file_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "tokenizer.json"
            path.write_bytes(b"synthetic")
            manifest = {"files": [{"path": "tokenizer.json", "bytes": 9, "sha256": hashlib.sha256(b"synthetic").hexdigest()}]}
            SIDE.verify_assets(directory, manifest)
            path.write_bytes(b"different")
            with self.assertRaises(ValueError):
                SIDE.verify_assets(directory, manifest)
            path.unlink()
            with self.assertRaises(ValueError):
                SIDE.verify_assets(directory, manifest)


class HTTPControls(unittest.TestCase):
    def setUp(self):
        self.server = SIDE.Server(("127.0.0.1", 0), make_engine())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def call(self, path, value=None, *, headers=None, method="POST"):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
        try:
            raw = json.dumps(value).encode() if value is not None else None
            connection.request(method, path, raw, headers or {"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_health_and_compatible_alias_routes(self):
        code, health = self.call("/health", method="GET")
        self.assertEqual(code, 200)
        self.assertEqual(health["dimensions"], 384)
        self.assertFalse(health["runtime_downloads"])
        for path in ["/v1/embeddings", "/embeddings"]:
            code, result = self.call(path, body())
            self.assertEqual(code, 200)
            self.assertEqual(result["model"], SIDE.MODEL)
            self.assertEqual(len(result["data"][0]["embedding"]), 384)

    def test_oversized_declared_body_rejected_without_read(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
        try:
            connection.putrequest("POST", "/v1/embeddings")
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", str(SIDE.MAX_BODY_BYTES + 1))
            connection.endheaders()
            self.assertEqual(connection.getresponse().status, 413)
        finally:
            connection.close()

    def test_duplicate_content_length_and_chunked_requests_refused(self):
        for chunked in [False, True]:
            connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
            try:
                connection.putrequest("POST", "/v1/embeddings")
                connection.putheader("Content-Type", "application/json")
                connection.putheader("Content-Length", "2")
                connection.putheader("Transfer-Encoding" if chunked else "Content-Length", "chunked" if chunked else "2")
                connection.endheaders()
                self.assertEqual(connection.getresponse().status, 400)
            finally:
                connection.close()

    def test_unknown_route_and_request_field_rejected(self):
        self.assertEqual(self.call("/public-memory", body())[0], 404)
        code, result = self.call("/v1/embeddings", {**body(), "credential": "secret-test-value"})
        self.assertEqual(code, 400)
        self.assertNotIn("secret-test-value", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
