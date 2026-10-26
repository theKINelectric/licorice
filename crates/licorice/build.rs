//! Compiles sqlite-vector into this crate so it links against the same SQLite that
//! rusqlite bundles (SPEC §7.5).
//!
//! The extension source comes from `vendor/sqlite-vector` (a vendored copy of tag
//! 1.1.2), or from `SQLITE_VECTOR_DIR` when set. It is compiled with
//! `-DSQLITE_CORE` against libsqlite3-sys's header, which Cargo exposes as
//! `DEP_SQLITE3_INCLUDE` only because `libsqlite3-sys` is a direct dependency.
//! The vendored `libs/sqlite3.c` is never compiled: one process, one SQLite.

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = match env::var("SQLITE_VECTOR_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => manifest.join("../../vendor/sqlite-vector"),
    };
    let src = root.join("src");
    if !src.join("sqlite-vector.c").exists() {
        panic!(
            "sqlite-vector source not found at {}; restore vendor/sqlite-vector or set SQLITE_VECTOR_DIR",
            src.display()
        );
    }
    let sqlite_include = env::var("DEP_SQLITE3_INCLUDE")
        .expect("DEP_SQLITE3_INCLUDE missing: libsqlite3-sys must be a direct dependency with `bundled`");
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let msvc = env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");

    let mut common = cc::Build::new();
    common
        .define("SQLITE_CORE", None)
        // libsqlite3-sys's header comes first so `#include "sqlite3.h"` never picks up the
        // vendored copy in libs/, which is only on the path for `fp16/fp16.h`.
        .include(&sqlite_include)
        .include(&src)
        .include(root.join("libs"))
        .opt_level(3)
        .warnings(false);

    // Portable kernels; each file guards itself on the target's ISA macros.
    let mut core = common.clone();
    for file in ["sqlite-vector.c", "distance-cpu.c", "distance-sse2.c", "distance-neon.c", "distance-rvv.c"] {
        core.file(src.join(file));
    }
    core.compile("sqlite_vector_core");

    // x86 wide kernels are compiled with their ISA enabled per translation unit,
    // mirroring the upstream Makefile; the extension picks a backend at runtime.
    if arch == "x86_64" {
        let mut avx2 = common.clone();
        if msvc {
            avx2.flag("/arch:AVX2");
        } else {
            avx2.flag("-mavx2").flag("-mfma");
        }
        avx2.file(src.join("distance-avx2.c")).compile("sqlite_vector_avx2");

        let mut avx512 = common.clone();
        if msvc {
            avx512.flag("/arch:AVX512");
        } else {
            for flag in ["-mavx512f", "-mavx512bw", "-mavx512vl", "-mavx512dq"] {
                avx512.flag(flag);
            }
        }
        avx512.file(src.join("distance-avx512.c")).compile("sqlite_vector_avx512");
    }

    println!("cargo:rerun-if-env-changed=SQLITE_VECTOR_DIR");
    println!("cargo:rerun-if-changed={}", src.display());
    println!("cargo:rerun-if-changed=build.rs");
}
