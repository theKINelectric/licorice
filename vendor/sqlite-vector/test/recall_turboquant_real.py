#!/usr/bin/env python3
"""Recall benchmark for TurboQuant on a real ANN-Benchmarks dataset."""

import argparse
import os
import sqlite3
import sys
import tempfile
import time
import urllib.request


DATASET_URLS = (
    "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
    "https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/fashion-mnist-784-euclidean.hdf5",
)
DATASET_NAME = "fashion-mnist-784-euclidean.hdf5"


def require_deps():
    try:
        import h5py  # noqa: F401
        import numpy as np  # noqa: F401
    except ImportError as exc:
        print(f"missing Python dependency: {exc}", file=sys.stderr)
        print("install with: python3 -m pip install h5py numpy", file=sys.stderr)
        sys.exit(2)


def default_extension_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("vector.dylib", "vector.so", "vector.dll", "vector"):
        path = os.path.join(root, "dist", name)
        if os.path.exists(path):
            return path
    return os.path.join(root, "dist", "vector")


def download_dataset(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, DATASET_NAME)
    if not os.path.exists(path):
        last_error = None
        for url in DATASET_URLS:
            try:
                print(f"downloading {url}")
                req = urllib.request.Request(url, headers={"User-Agent": "sqlite-vector-recall/1.0"})
                with urllib.request.urlopen(req) as response, open(path, "wb") as out:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                try:
                    os.remove(path)
                except OSError:
                    pass
        if last_error is not None:
            raise last_error
    return path


def blob(row):
    import numpy as np

    return sqlite3.Binary(np.ascontiguousarray(row, dtype="<f4").tobytes())


def ids_for(conn, sql, query):
    return [row[0] for row in conn.execute(sql, (blob(query),))]


def recall_at_k(exact, approx):
    total = 0
    for lhs, rhs in zip(exact, approx):
        total += len(set(lhs).intersection(rhs))
    return total / float(len(exact) * len(exact[0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/sqlite-vector"))
    parser.add_argument("--extension", default=default_extension_path())
    parser.add_argument("--vectors", type=int, default=10000)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--qbits", default="2,3,4")
    parser.add_argument("--no-preload", action="store_true")
    args = parser.parse_args()

    require_deps()
    import h5py

    dataset_path = download_dataset(args.cache_dir)
    with h5py.File(dataset_path, "r") as h5:
        train = h5["train"][: args.vectors]
        test = h5["test"][: args.queries]

    dim = int(train.shape[1])
    qbits_values = [int(part) for part in args.qbits.split(",") if part]

    fd, db_path = tempfile.mkstemp(prefix="sqlite_vector_recall_", suffix=".db")
    os.close(fd)
    try:
        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)
        conn.load_extension(args.extension)
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA cache_size=-20000")
        conn.execute("CREATE TABLE items(id INTEGER PRIMARY KEY, v BLOB)")

        t0 = time.perf_counter()
        with conn:
            conn.executemany(
                "INSERT INTO items(id, v) VALUES(?1, ?2)",
                ((i + 1, blob(train[i])) for i in range(len(train))),
            )
        insert_ms = (time.perf_counter() - t0) * 1000.0

        conn.execute(
            "SELECT vector_init('items', 'v', ?1)",
            (f"type=f32,dimension={dim},distance=L2",),
        ).fetchone()

        full_sql = f"SELECT id FROM vector_full_scan('items', 'v', ?1, {args.k})"
        t0 = time.perf_counter()
        exact = [ids_for(conn, full_sql, query) for query in test]
        exact_ms = (time.perf_counter() - t0) * 1000.0

        distance_backend, turbo_backend = conn.execute(
            "SELECT vector_backend(), vector_turboquant_backend()"
        ).fetchone()
        print(
            f"dataset={DATASET_NAME} vectors={len(train)} dim={dim} "
            f"queries={len(test)} k={args.k} distance=L2"
        )
        print(f"backend distance={distance_backend} turboquant={turbo_backend}")
        print(f"insert_ms={insert_ms:.3f} exact_ms={exact_ms:.3f} exact_per_query_ms={exact_ms / len(test):.3f}")

        for qbits in qbits_values:
            t0 = time.perf_counter()
            conn.execute(
                "SELECT vector_quantize('items', 'v', ?1)",
                (f"qtype=TURBO,qbits={qbits},max_memory=0",),
            ).fetchone()
            quant_ms = (time.perf_counter() - t0) * 1000.0

            if not args.no_preload:
                conn.execute("SELECT vector_quantize_preload('items', 'v')").fetchone()

            storage_bytes = conn.execute("SELECT vector_quantize_memory('items', 'v')").fetchone()[0]
            turbo_sql = f"SELECT id FROM vector_quantize_scan('items', 'v', ?1, {args.k})"
            t0 = time.perf_counter()
            approx = [ids_for(conn, turbo_sql, query) for query in test]
            turbo_ms = (time.perf_counter() - t0) * 1000.0
            recall = recall_at_k(exact, approx)
            print(
                f"qbits={qbits} preload={int(not args.no_preload)} "
                f"build_ms={quant_ms:.3f} storage_bytes={storage_bytes} "
                f"turbo_ms={turbo_ms:.3f} turbo_per_query_ms={turbo_ms / len(test):.3f} "
                f"speedup={exact_ms / turbo_ms:.2f}x recall@{args.k}={recall:.4f}"
            )

        conn.close()
    finally:
        try:
            os.remove(db_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
