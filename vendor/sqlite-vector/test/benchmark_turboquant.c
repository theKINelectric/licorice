/*
 * benchmark_turboquant.c
 *
 * Synthetic brute-force vs TurboQuant benchmark for sqlite-vector.
 * Build example:
 *   gcc -DSQLITE_CORE -O3 -Isrc -Ilibs test/benchmark_turboquant.c libs/sqlite3.c src/sqlite-vector.c ... -o build/benchmark_turboquant -lm -lpthread
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sqlite3.h"
#include "sqlite-vector.h"

#ifndef NVECS
#define NVECS 4000
#endif
#ifndef NQUERIES
#define NQUERIES 40
#endif
#ifndef DIM
#define DIM 128
#endif
#ifndef K
#define K 10
#endif
#ifndef Q_BITS
#define Q_BITS 4
#endif
#ifndef KEEP_VECTORS
#define KEEP_VECTORS 1
#endif
#ifndef DB_PATH
#define DB_PATH ":memory:"
#endif
#ifndef SQLITE_CACHE_KB
#define SQLITE_CACHE_KB 65536
#endif
#ifndef PRELOAD
#define PRELOAD 1
#endif

static uint64_t rng_state = 0x123456789abcdef0ull;

static void normalize(float *v, int dim);

static uint64_t splitmix64(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static float rand_f32(void) {
    return (float)((double)(splitmix64() >> 11) * (1.0 / 9007199254740992.0));
}

static float rand_normal(void) {
    float u1 = rand_f32();
    float u2 = rand_f32();
    if (u1 < 1e-7f) u1 = 1e-7f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530717958647692f * u2);
}

static uint64_t vector_seed(int idx) {
    uint64_t x = 0x9E3779B97F4A7C15ull ^ ((uint64_t)(idx + 1) * 0xBF58476D1CE4E5B9ull);
    x ^= (uint64_t)DIM * 0x94D049BB133111EBull;
    return x;
}

static void fill_vector_for_id(int idx, float *row) {
    uint64_t saved = rng_state;
    rng_state = vector_seed(idx);
    for (int j = 0; j < DIM; ++j) row[j] = rand_normal();
    normalize(row, DIM);
    rng_state = saved;
}

static void normalize(float *v, int dim) {
    double norm_sq = 0.0;
    for (int i = 0; i < dim; ++i) norm_sq += (double)v[i] * (double)v[i];
    float inv = norm_sq > 1e-20 ? 1.0f / sqrtf((float)norm_sq) : 0.0f;
    for (int i = 0; i < dim; ++i) v[i] *= inv;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static int exec_or_die(sqlite3 *db, const char *sql) {
    char *err = NULL;
    int rc = sqlite3_exec(db, sql, NULL, NULL, &err);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\nSQL: %s\n", err ? err : sqlite3_errmsg(db), sql);
        sqlite3_free(err);
        exit(1);
    }
    return rc;
}

static void run_query(sqlite3_stmt *stmt, const float *query, int ids[K]) {
    sqlite3_reset(stmt);
    sqlite3_clear_bindings(stmt);
    sqlite3_bind_blob(stmt, 1, query, DIM * (int)sizeof(float), SQLITE_STATIC);

    int i = 0;
    while (sqlite3_step(stmt) == SQLITE_ROW && i < K) {
        ids[i++] = sqlite3_column_int(stmt, 0);
    }
    while (i < K) ids[i++] = -1;
}

static int overlap_at_k(const int a[K], const int b[K]) {
    int count = 0;
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < K; ++j) {
            if (a[i] == b[j]) {
                count++;
                break;
            }
        }
    }
    return count;
}

int main(void) {
    sqlite3 *db = NULL;
    if (strcmp(DB_PATH, ":memory:") != 0) remove(DB_PATH);
    if (sqlite3_open(DB_PATH, &db) != SQLITE_OK) {
        fprintf(stderr, "cannot open sqlite database\n");
        return 1;
    }
    if (sqlite3_vector_init(db, NULL, NULL) != SQLITE_OK) {
        fprintf(stderr, "cannot init sqlite-vector\n");
        return 1;
    }

    float *vectors = KEEP_VECTORS ? (float *)malloc((size_t)NVECS * DIM * sizeof(float)) : NULL;
    float *scratch = KEEP_VECTORS ? NULL : (float *)malloc((size_t)DIM * sizeof(float));
    float *queries = (float *)malloc((size_t)NQUERIES * DIM * sizeof(float));
    int (*exact_ids)[K] = malloc((size_t)NQUERIES * sizeof(*exact_ids));
    int (*turbo_ids)[K] = malloc((size_t)NQUERIES * sizeof(*turbo_ids));
    if ((KEEP_VECTORS && !vectors) || (!KEEP_VECTORS && !scratch) || !queries || !exact_ids || !turbo_ids) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }

    if (KEEP_VECTORS) {
        for (int i = 0; i < NVECS; ++i) {
            float *row = vectors + (size_t)i * DIM;
            fill_vector_for_id(i, row);
        }
    }
    for (int i = 0; i < NQUERIES; ++i) {
        int base = (int)(splitmix64() % NVECS);
        float *q = queries + (size_t)i * DIM;
        float *v = KEEP_VECTORS ? vectors + (size_t)base * DIM : scratch;
        if (!KEEP_VECTORS) fill_vector_for_id(base, scratch);
        for (int j = 0; j < DIM; ++j) q[j] = v[j] + 0.03f * rand_normal();
        normalize(q, DIM);
    }

    exec_or_die(db, "PRAGMA journal_mode=OFF;");
    exec_or_die(db, "PRAGMA synchronous=OFF;");
    char pragma_sql[128];
    snprintf(pragma_sql, sizeof(pragma_sql), "PRAGMA cache_size=-%d;", SQLITE_CACHE_KB);
    exec_or_die(db, pragma_sql);
    exec_or_die(db, "CREATE TABLE bench(id INTEGER PRIMARY KEY, v BLOB);");
    sqlite3_stmt *insert = NULL;
    sqlite3_prepare_v2(db, "INSERT INTO bench(id, v) VALUES(?1, ?2);", -1, &insert, NULL);
    for (int i = 0; i < NVECS; ++i) {
        sqlite3_reset(insert);
        sqlite3_clear_bindings(insert);
        sqlite3_bind_int(insert, 1, i + 1);
        const float *row = KEEP_VECTORS ? vectors + (size_t)i * DIM : scratch;
        if (!KEEP_VECTORS) fill_vector_for_id(i, scratch);
        sqlite3_bind_blob(insert, 2, row, DIM * (int)sizeof(float), SQLITE_STATIC);
        if (sqlite3_step(insert) != SQLITE_DONE) {
            fprintf(stderr, "insert failed: %s\n", sqlite3_errmsg(db));
            return 1;
        }
    }
    sqlite3_finalize(insert);

    char init_sql[256];
    snprintf(init_sql, sizeof(init_sql), "SELECT vector_init('bench', 'v', 'type=f32,dimension=%d,distance=DOT');", DIM);
    exec_or_die(db, init_sql);

    double t0 = now_ms();
    char quant_sql[128];
    snprintf(quant_sql, sizeof(quant_sql), "SELECT vector_quantize('bench', 'v', 'qtype=TURBO,qbits=%d,max_memory=0');", Q_BITS);
    exec_or_die(db, quant_sql);
    double quant_ms = now_ms() - t0;
#if PRELOAD
    exec_or_die(db, "SELECT vector_quantize_preload('bench', 'v');");
#endif

    sqlite3_stmt *full = NULL;
    sqlite3_stmt *turbo = NULL;
    sqlite3_prepare_v2(db, "SELECT id FROM vector_full_scan('bench', 'v', ?1, 10);", -1, &full, NULL);
    sqlite3_prepare_v2(db, "SELECT id FROM vector_quantize_scan('bench', 'v', ?1, 10);", -1, &turbo, NULL);

    t0 = now_ms();
    for (int i = 0; i < NQUERIES; ++i) run_query(full, queries + (size_t)i * DIM, exact_ids[i]);
    double full_ms = now_ms() - t0;

    t0 = now_ms();
    for (int i = 0; i < NQUERIES; ++i) run_query(turbo, queries + (size_t)i * DIM, turbo_ids[i]);
    double turbo_ms = now_ms() - t0;

    int overlap = 0;
    for (int i = 0; i < NQUERIES; ++i) overlap += overlap_at_k(exact_ids[i], turbo_ids[i]);
    double recall = (double)overlap / (double)(NQUERIES * K);

    sqlite3_finalize(full);
    sqlite3_finalize(turbo);

    int open_stmts = 0;
    for (sqlite3_stmt *stmt = sqlite3_next_stmt(db, NULL); stmt; stmt = sqlite3_next_stmt(db, stmt)) open_stmts++;

    sqlite3_int64 memory = 0;
    sqlite3_stmt *mem = NULL;
    sqlite3_prepare_v2(db, "SELECT vector_quantize_memory('bench', 'v');", -1, &mem, NULL);
    if (sqlite3_step(mem) == SQLITE_ROW) memory = sqlite3_column_int64(mem, 0);
    sqlite3_finalize(mem);

    const unsigned char *distance_backend = (const unsigned char *)"unknown";
    const unsigned char *turbo_backend = (const unsigned char *)"unknown";
    sqlite3_stmt *backend = NULL;
    if (sqlite3_prepare_v2(db, "SELECT vector_backend(), vector_turboquant_backend();", -1, &backend, NULL) == SQLITE_OK &&
        sqlite3_step(backend) == SQLITE_ROW) {
        const unsigned char *v = sqlite3_column_text(backend, 0);
        if (v) distance_backend = v;
        v = sqlite3_column_text(backend, 1);
        if (v) turbo_backend = v;
    }

    printf("dataset vectors=%d dim=%d queries=%d k=%d qbits=%d preload=%d\n", NVECS, DIM, NQUERIES, K, Q_BITS, PRELOAD);
    printf("backend distance=%s turboquant=%s\n", distance_backend, turbo_backend);
    printf("turboquant build_ms=%.3f storage_bytes=%lld\n", quant_ms, (long long)memory);
    printf("full_scan_ms=%.3f per_query_ms=%.3f\n", full_ms, full_ms / NQUERIES);
    printf("turboquant_ms=%.3f per_query_ms=%.3f speedup=%.2fx\n", turbo_ms, turbo_ms / NQUERIES, full_ms / turbo_ms);
    printf("recall@%d=%.4f open_statements=%d\n", K, recall, open_stmts);

    if (backend) sqlite3_finalize(backend);
    free(vectors);
    free(scratch);
    free(queries);
    free(exact_ids);
    free(turbo_ids);
    sqlite3_close(db);
    if (strcmp(DB_PATH, ":memory:") != 0) remove(DB_PATH);
    return open_stmts == 0 ? 0 : 1;
}
