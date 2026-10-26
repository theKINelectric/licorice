# sqlite_vector

SQLite Vector is a cross-platform, ultra-efficient SQLite extension that brings vector search capabilities to your embedded database. It works seamlessly on iOS, Android, Windows, Linux, and macOS, using just 30MB of memory by default. With support for Float32, Float16, BFloat16, Int8, UInt8 and 1Bit, and highly optimized distance functions, it's the ideal solution for Edge AI applications.

## Installation

```
dart pub add sqlite_vector
```

Requires Dart 3.10+ / Flutter 3.38+.

## Usage

### With `sqlite3`

```dart
import 'package:sqlite3/sqlite3.dart';
import 'package:sqlite_vector/sqlite_vector.dart';

void main() {
  // Load once at startup.
  sqlite3.loadSqliteVectorExtension();

  final db = sqlite3.openInMemory();

  // Create a regular table with a BLOB column for vectors.
  db.execute('CREATE TABLE items (id INTEGER PRIMARY KEY, embedding BLOB)');

  // Insert vectors.
  final stmt = db.prepare('INSERT INTO items (embedding) VALUES (vector_as_f32(?))');
  stmt.execute(['[1.0, 2.0, 3.0, 4.0]']);
  stmt.dispose();

  // Initialize the vector index.
  db.execute("SELECT vector_init('items', 'embedding', 'type=FLOAT32,dimension=4')");

  // Find the 2 nearest neighbors.
  final results = db.select('''
    SELECT e.id, v.distance FROM items AS e
    JOIN vector_full_scan('items', 'embedding', vector_as_f32('[1.0, 2.0, 3.0, 4.0]'), 2) AS v
    ON e.id = v.rowid
  ''');

  db.dispose();
}
```

### With `drift`

```dart
import 'package:sqlite3/sqlite3.dart';
import 'package:sqlite_vector/sqlite_vector.dart';
import 'package:drift/native.dart';

Sqlite3 loadExtensions() {
  sqlite3.loadSqliteVectorExtension();
  return sqlite3;
}

// Use when creating the database:
NativeDatabase.createInBackground(
  File(path),
  sqlite3: loadExtensions,
);
```

## Supported platforms

| Platform | Architectures |
|----------|---------------|
| Android  | arm64, arm, x64 |
| iOS      | arm64 (device + simulator) |
| macOS    | arm64, x64 |
| Linux    | arm64, x64 |
| Windows  | x64 |

## API

See the full [sqlite-vector API documentation](https://github.com/sqliteai/sqlite-vector/blob/main/API.md).

## License

See [LICENSE](LICENSE).
