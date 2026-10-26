//! Opening connections: loading sqlite-vector, pragmas, version check, `vector_init`.

use std::ffi::{CStr, c_char, c_int};
use std::path::Path;

use rusqlite::{Connection, OpenFlags, ffi, params};

use super::vector::VectorKind;
use crate::{Error, Result};

/// The sqlite-vector release this build is pinned to (`vendor/sqlite-vector`).
pub(crate) const SQLITE_VECTOR_VERSION: &str = "1.1.2";

unsafe extern "C" {
    fn sqlite3_vector_init(
        db: *mut ffi::sqlite3,
        err: *mut *mut c_char,
        api: *const ffi::sqlite3_api_routines,
    ) -> c_int;
}

/// Loads sqlite-vector into one connection; idempotent. Licorice loads it into its own
/// connections only. It is not an auto-extension, which would reach every SQLite
/// database the host process opens (and write a `_sqliteai_vector` table into each).
/// Call this to use `vector_*` functions on a connection you opened yourself.
pub fn load_vector_extension(conn: &Connection) -> Result<()> {
    let mut err: *mut c_char = std::ptr::null_mut();
    // SAFETY: the handle belongs to an open connection that outlives the call. Built
    // with SQLITE_CORE, the entry point never reads `api`. A non-null `err` was
    // allocated by SQLite and is freed here, once.
    let rc = unsafe { sqlite3_vector_init(conn.handle(), &mut err, std::ptr::null()) };
    if rc == ffi::SQLITE_OK {
        return Ok(());
    }
    let message = if err.is_null() {
        format!("error code {rc}")
    } else {
        // SAFETY: SQLite returns a NUL-terminated string.
        let message = unsafe { CStr::from_ptr(err) }.to_string_lossy().into_owned();
        unsafe { ffi::sqlite3_free(err.cast()) };
        message
    };
    Err(Error::Extension(message))
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Role {
    Writer,
    Reader,
}

/// Opens a connection with Licorice's pragmas and checks the linked sqlite-vector version.
pub(crate) fn open(path: &Path, role: Role) -> Result<Connection> {
    let flags = match role {
        Role::Writer => OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
        // Readers open read-write so sqlite-vector can initialize, then refuse writes.
        Role::Reader => OpenFlags::SQLITE_OPEN_READ_WRITE,
    } | OpenFlags::SQLITE_OPEN_NO_MUTEX
        | OpenFlags::SQLITE_OPEN_URI;
    let conn = Connection::open_with_flags(path, flags)?;
    load_vector_extension(&conn)?;
    if role == Role::Writer {
        conn.pragma_update(None, "journal_mode", "WAL")?;
    }
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    conn.pragma_update(None, "busy_timeout", 5000)?;
    conn.pragma_update(None, "temp_store", "MEMORY")?;
    conn.pragma_update(None, "mmap_size", 256i64 * 1024 * 1024)?;
    if role == Role::Reader {
        conn.pragma_update(None, "query_only", "ON")?;
    }
    let found: String = conn.query_row("SELECT vector_version()", [], |r| r.get(0))?;
    if found != SQLITE_VECTOR_VERSION {
        return Err(Error::ExtensionVersion { expected: SQLITE_VECTOR_VERSION, found });
    }
    Ok(conn)
}

/// `vector_init` options for every vector table: FLOAT32 vectors, L2-normalized in Rust
/// before insert, so cosine distance can be computed as `1 - dot` on exact scans.
pub(crate) fn vector_options(dim: usize) -> String {
    format!("type=FLOAT32,dimension={dim},distance=COSINE,normalized=1")
}

/// Runs `vector_init` for every vector table and returns the quantization generation of
/// each, read in the same snapshot.
///
/// sqlite-vector keeps its per-table state (including quantization type, scale and
/// offset) in a per-connection context that is filled only when `vector_init` first
/// runs. A connection therefore stays tied to the generation it was initialized at;
/// the reader pool replaces connections that fall behind.
pub(crate) fn init_vectors(conn: &Connection, dim: usize) -> Result<[i64; VectorKind::COUNT]> {
    let tx = conn.unchecked_transaction()?;
    let mut generations = [0i64; VectorKind::COUNT];
    {
        let mut stmt = tx.prepare("SELECT generation FROM vec_state WHERE tbl = ?1")?;
        for kind in VectorKind::ALL {
            generations[kind.index()] = stmt.query_row([kind.table()], |r| r.get(0))?;
        }
    }
    let opts = vector_options(dim);
    for kind in VectorKind::ALL {
        tx.query_row("SELECT vector_init(?1, 'v', ?2)", params![kind.table(), opts], |_| Ok(()))?;
    }
    tx.commit()?;
    Ok(generations)
}
