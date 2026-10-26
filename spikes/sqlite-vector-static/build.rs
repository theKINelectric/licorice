use std::{env, path::PathBuf};
fn main() {
    let root = PathBuf::from(env::var("SQLITE_VECTOR_DIR").expect("SQLITE_VECTOR_DIR"));
    let src = root.join("src");
    // Header from libsqlite3-sys (bundled) so we compile against the exact SQLite we link.
    let sqlite_inc = env::var("DEP_SQLITE3_INCLUDE").expect("DEP_SQLITE3_INCLUDE");
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let mut common = cc::Build::new();
    common.define("SQLITE_CORE", None).include(&sqlite_inc).include(&src)
        .include(root.join("libs/fp16")).include(root.join("libs")).opt_level(3).warnings(false);
    // core + portable kernels
    let mut b = common.clone();
    for f in ["sqlite-vector.c", "distance-cpu.c", "distance-sse2.c", "distance-neon.c", "distance-rvv.c"] { b.file(src.join(f)); }
    b.compile("sqlite_vector_core");
    if target_arch == "x86_64" {
        let mut avx2 = common.clone(); avx2.flag("-mavx2").flag("-mfma").file(src.join("distance-avx2.c")).compile("sqlite_vector_avx2");
        let mut avx512 = common.clone();
        for fl in ["-mavx512f", "-mavx512bw", "-mavx512vl", "-mavx512dq"] { avx512.flag(fl); }
        avx512.file(src.join("distance-avx512.c")).compile("sqlite_vector_avx512");
    }
    println!("cargo:rerun-if-changed=build.rs");
}
