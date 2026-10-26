#!/usr/bin/env python3
"""Private, whole-identity backup/restore. Standard library; no provider calls."""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sqlite3
import stat
import subprocess
import tempfile
import time
import zipfile

SCHEMA = 11
FORMAT = 'licorice-memory-backup'
MAX_DATABASE = 256 * 1024 * 1024
MAX_METADATA = 64 * 1024
MAX_EVENTS = 100000
MAX_NATIVE_JSON = 32 * 1024 * 1024
QUERY = 'current commitments, methods and changed judgments'


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_safe_sqlite():
    require(sqlite3.sqlite_version_info >= (3, 53, 2),
            'Python SQLite must be >=3.53.2 for memory file processing; this Python uses '
            + sqlite3.sqlite_version + '. Select an updated Python interpreter.')


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}', value), 'Invalid explicit identity/scope')
    return value


def absolute(value, leaf=False):
    path = Path(value).expanduser()
    require(path.is_absolute(), 'Use an absolute path')
    if leaf:
        path = path.parent.resolve(strict=True) / path.name
        require(not path.is_symlink(), 'Refuse destination symlink')
        require(path.parent.is_dir(), 'Destination parent must exist')
        return path
    return path.resolve(strict=True)


def load_binding(path):
    data = json.loads(absolute(path).read_text())
    require(isinstance(data, dict), 'Binding must be an object')
    if 'LICORICE_BEING_AGENT' in data:
        require(data.get('LICORICE_BEING_BACKEND') == 'native', 'Expected a native binding')
        data = {'agent_id': data['LICORICE_BEING_AGENT'], 'binary': data['LICORICE_BEING_BIN'],
                'store': data['LICORICE_BEING_STORE'], 'embed_dim': int(data['LICORICE_BEING_DIM']),
                'relationship_id': data.get('LICORICE_BEING_RELATIONSHIP'),
                'include_episodes': data.get('LICORICE_BEING_EPISODES', '0') == '1'}
    else:
        require(data.get('version') == 1 and data.get('backend') == 'native', 'Expected a version-1 native binding')
    identifier(data['agent_id'])
    if data.get('relationship_id') is not None:
        identifier(data['relationship_id'])
    require(type(data['embed_dim']) is int and 1 <= data['embed_dim'] <= 65536, 'Invalid embedding dimension')
    require(type(data.get('include_episodes', False)) is bool, 'Episode selection must be boolean')
    binary = absolute(data['binary'])
    require(binary.is_file() and os.access(binary, os.X_OK), 'Explicit native executable is required')
    return {**data, 'binary': str(binary), 'store': str(absolute(data['store'])),
            'include_episodes': data.get('include_episodes', False), 'relationship_id': data.get('relationship_id')}


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_private(path, content):
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def consistent_copy(source, destination):
    require_safe_sqlite()
    require(source.is_file() and source.stat().st_size <= MAX_DATABASE, 'Source must be a native store of at most 256 MiB')
    deadline = time.monotonic() + 30
    with closing(sqlite3.connect(source.as_uri() + '?mode=ro', uri=True, timeout=5)) as src:
        src.execute('PRAGMA query_only=ON')
        src.execute('PRAGMA trusted_schema=OFF')
        page_size = src.execute('PRAGMA page_size').fetchone()[0]
        def progress(status, remaining, total):
            require(time.monotonic() <= deadline and total * page_size <= MAX_DATABASE, 'Backup copy exceeded its time/size bound')
        with closing(sqlite3.connect(destination)) as dst:
            src.backup(dst, pages=256, progress=progress, sleep=0.01)
    destination.chmod(0o600)
    require(destination.stat().st_size <= MAX_DATABASE, 'Snapshot exceeds 256 MiB')


def describe(database, expected_agent, expected_dim):
    require_safe_sqlite()
    require(0 < database.stat().st_size <= MAX_DATABASE, 'Invalid database size')
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('PRAGMA query_only=ON'); db.execute('PRAGMA trusted_schema=OFF')
        require(db.execute('PRAGMA user_version').fetchone()[0] == SCHEMA, 'Requires schema11; no automatic upgrade/downgrade')
        require(db.execute('PRAGMA quick_check').fetchall() == [('ok',)], 'SQLite integrity check failed')
        require(db.execute('SELECT agent_id FROM being_owner WHERE singleton=1').fetchone() == (expected_agent,), 'Store owner differs from expected agent')
        require(db.execute("SELECT value FROM meta WHERE key='embed_dim'").fetchone() == (str(expected_dim),), 'Embedding dimension mismatch')
        count = db.execute('SELECT count(*) FROM being_events').fetchone()[0]
        require(count <= MAX_EVENTS, 'Backup supports at most 100000 events')
        digest = hashlib.sha256()
        for seq, row in enumerate(db.execute('SELECT seq,actor,request,digest,recorded_at,imported_from FROM being_events ORDER BY seq'), 1):
            require(row[0] == seq, 'Noncontiguous journal refused')
            digest.update(canonical(row) + b'\n')
    return {'revision': count, 'events': count, 'journal_sha256': digest.hexdigest()}


def native(binary, command, database, binding, directory, request=None):
    args = [str(binary), command, '--store', str(database), '--agent', binding['agent_id'],
            '--embed-dim', str(binding['embed_dim'])]
    if binding.get('relationship_id'):
        args += ['--relationship', binding['relationship_id']]
    with tempfile.TemporaryFile(dir=directory) as stdout, tempfile.TemporaryFile(dir=directory) as stderr:
        proc = subprocess.run(args, input=canonical(request) if request else None,
                              stdout=stdout, stderr=stderr, timeout=30)
        require(stdout.tell() <= MAX_NATIVE_JSON, 'Native validation response exceeds 32 MiB')
        stdout.seek(0)
        try:
            response = json.load(stdout)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError('Selected native runtime did not return JSON') from exc
        require(proc.returncode == 0 and response.get('ok') is True, 'Selected native runtime refused validation')
    return response['result']


def validate_native(database, binding, directory):
    # Store::open may write/migrate. It only ever receives another disposable copy.
    check = directory / 'native-check.sqlite'
    shutil.copyfile(database, check); check.chmod(0o600)
    snapshot = native(binding['binary'], 'snapshot', check, binding, directory)
    context = native(binding['binary'], 'context', check, binding, directory, {
        'query': QUERY, 'budget_tokens': 2000, 'include_episodes': binding['include_episodes']})
    require(snapshot['agent_id'] == binding['agent_id'] and snapshot['revision'] == context['continuity']['revision'], 'Native identity/revision disagreement')
    usage = context['usage']
    require(usage['budget_tokens'] == 2000 and 0 <= usage['tokens'] <= 2000
            and usage['tokenizer'] == 'cl100k_base:ordinary'
            and context['evidence_mode'] == 'continuity_only', 'Native continuity cap/counting contract failed')
    return {'snapshot_sha256': hashlib.sha256(canonical(snapshot)).hexdigest(),
            'native_revision': snapshot['revision'],
            'context': {'query': QUERY, 'budget_tokens': 2000, 'tokens': usage['tokens'],
                        'text_sha256': hashlib.sha256(context['text'].encode()).hexdigest()}}


def backup(binding, output):
    output = absolute(output, leaf=True)
    require(not output.exists(), 'Backup output already exists')
    with tempfile.TemporaryDirectory(prefix='.licorice-backup-', dir=output.parent) as work:
        work = Path(work); database = work / 'memory.sqlite'
        consistent_copy(Path(binding['store']), database)
        facts = describe(database, binding['agent_id'], binding['embed_dim'])
        native_proof = validate_native(database, binding, work)
        require(native_proof['native_revision'] == facts['revision'], 'Database and native revision disagree')
        metadata = {'format': FORMAT, 'version': 1, 'schema_version': SCHEMA,
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'agent_id': binding['agent_id'], 'embed_dim': binding['embed_dim'],
                    'relationship_id': binding.get('relationship_id'), 'include_episodes': binding['include_episodes'],
                    **facts, 'snapshot_sha256': native_proof['snapshot_sha256'], 'context': native_proof['context'],
                    'native_binary_sha256': sha(binding['binary']),
                    'database': {'name': 'memory.sqlite', 'size_bytes': database.stat().st_size, 'sha256': sha(database)}}
        encoded = canonical(metadata)
        require(len(encoded) <= MAX_METADATA, 'Envelope metadata exceeds 64 KiB')
        archive = work / 'backup.licorice'
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED) as zip:
            zip.writestr('manifest.json', encoded)
            zip.write(database, 'memory.sqlite')
        archive.chmod(0o600)
        with archive.open('rb') as stream:
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and cannot replace a file or symlink.
        os.link(archive, output)
        sync_directory(output.parent)
    return {'format': FORMAT, 'version': 1, 'agent_id': metadata['agent_id'],
            'revision': facts['revision'], 'archive_sha256': sha(output), 'output': str(output)}


def unpack(archive, expected_agent, binary, work):
    identifier(expected_agent)
    binary = absolute(binary)
    require(binary.is_file() and os.access(binary, os.X_OK), 'Explicit native executable is required')
    archive = absolute(archive)
    require(archive.is_file() and archive.stat().st_size <= MAX_DATABASE + MAX_METADATA + 4096, 'Backup archive exceeds its bound')
    database = work / 'memory.sqlite'
    with zipfile.ZipFile(archive) as zip:
        members = zip.infolist()
        require(len(members) == 2 and {m.filename for m in members} == {'manifest.json', 'memory.sqlite'}, 'Expected exactly manifest.json and memory.sqlite')
        for member in members:
            mode = member.external_attr >> 16
            require(member.compress_type == zipfile.ZIP_STORED and not member.flag_bits & 1
                    and not member.is_dir() and stat.S_IFMT(mode) in (0, stat.S_IFREG), 'Compressed/encrypted/special members refused')
        require(zip.getinfo('manifest.json').file_size <= MAX_METADATA, 'Envelope metadata exceeds 64 KiB')
        metadata = json.loads(zip.read('manifest.json'))
        require(metadata['format'] == FORMAT and type(metadata['version']) is int and metadata['version'] == 1, 'Unsupported backup format/version')
        require(type(metadata['schema_version']) is int and metadata['schema_version'] == SCHEMA, 'Unsupported backup schema')
        require(metadata['agent_id'] == expected_agent, 'Backup owner differs from expected agent')
        require(type(metadata['revision']) is int and type(metadata['events']) is int
                and 0 <= metadata['revision'] == metadata['events'] <= MAX_EVENTS, 'Invalid journal count/revision')
        require(type(metadata['embed_dim']) is int and 1 <= metadata['embed_dim'] <= 65536, 'Invalid embedding dimension')
        require(type(metadata['include_episodes']) is bool, 'Invalid episode selection')
        if metadata['relationship_id'] is not None:
            identifier(metadata['relationship_id'])
        payload = metadata['database']
        require(payload['name'] == 'memory.sqlite' and type(payload['size_bytes']) is int
                and 0 < payload['size_bytes'] <= MAX_DATABASE
                and zip.getinfo('memory.sqlite').file_size == payload['size_bytes'], 'Invalid payload size/name')
        with zip.open('memory.sqlite') as source:
            with database.open('xb') as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
        database.chmod(0o600)
    require(database.stat().st_size == payload['size_bytes'] and sha(database) == payload['sha256'], 'Database payload digest mismatch')
    binding = {key: metadata[key] for key in ['agent_id', 'embed_dim', 'relationship_id', 'include_episodes']}
    binding['binary'] = str(binary)
    actual = describe(database, expected_agent, binding['embed_dim'])
    require(all(actual[key] == metadata[key] for key in actual), 'Journal revision/digest mismatch')
    proof = validate_native(database, binding, work)
    require(proof['snapshot_sha256'] == metadata['snapshot_sha256'] and proof['context'] == metadata['context'], 'Selected runtime changed restored snapshot/context; qualify compatibility explicitly')
    require(proof['native_revision'] == metadata['revision'], 'Native revision mismatch')
    return metadata, binding, database


def verify(archive, agent, binary):
    with tempfile.TemporaryDirectory(prefix='licorice-verify-backup-') as work:
        metadata, _, _ = unpack(archive, agent, binary, Path(work))
    return {'valid': True, 'agent_id': agent, 'revision': metadata['revision'],
            'schema_version': SCHEMA, 'context_tokens': metadata['context']['tokens'],
            'authentication': 'unkeyed_integrity_only', 'encryption': 'none'}


def restore(archive, agent, binary, destination):
    destination = absolute(destination, leaf=True)
    require(not destination.exists(), 'Restore destination must not exist, even if empty')
    with tempfile.TemporaryDirectory(prefix='.licorice-restore-', dir=destination.parent) as work:
        work = Path(work)
        metadata, selected, database = unpack(archive, agent, binary, work)
        state = work / 'state'; state.mkdir(mode=0o700)
        fd = os.open(state / 'memory.sqlite', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'wb') as target, database.open('rb') as source:
            shutil.copyfileobj(source, target, 1024 * 1024)
            target.flush(); os.fsync(target.fileno())
        binding = {'version': 1, 'backend': 'native', **selected,
                   'store': str(destination / 'memory.sqlite'), 'models': None}
        values = {'LICORICE_BEING_BACKEND': 'native', 'LICORICE_BEING_AGENT': agent,
                  'LICORICE_BEING_BIN': selected['binary'], 'LICORICE_BEING_STORE': binding['store'],
                  'LICORICE_BEING_DIM': str(selected['embed_dim']),
                  'LICORICE_BEING_EPISODES': '1' if selected['include_episodes'] else '0'}
        if selected['relationship_id'] is not None:
            values['LICORICE_BEING_RELATIONSHIP'] = selected['relationship_id']
        env = '\n'.join('export ' + key + '=' + shlex.quote(value) for key, value in values.items())
        if selected['relationship_id'] is None:
            env += '\nunset LICORICE_BEING_RELATIONSHIP'
        env += '\nunset LICORICE_BEING_MODELS\n'
        write_private(state / 'memory.env', env.encode())
        write_private(state / 'binding.json', canonical(binding) + b'\n')
        write_private(state / 'RESTORED.json', canonical({'format': FORMAT, 'version': 1,
                      'archive_sha256': sha(absolute(archive)), 'agent_id': agent,
                      'revision': metadata['revision'], 'database_sha256': metadata['database']['sha256'],
                      'binding_sha256': sha(state / 'binding.json')}) + b'\n')
        sync_directory(state)
        # Reserve a fresh name exclusively. POSIX rename replaces only this empty
        # reservation, publishing the whole validated tree in one operation.
        # A crash before rename leaves an empty, unbound directory, never half state.
        destination.mkdir(mode=0o700)
        reservation = destination.stat()
        require(not destination.is_symlink() and destination.stat().st_ino == reservation.st_ino
                and not any(destination.iterdir()), 'Restore reservation changed')
        os.rename(state, destination)
        sync_directory(destination.parent)
    return {'restored': True, 'agent_id': agent, 'revision': metadata['revision'],
            'binding': str(destination / 'binding.json'), 'environment': str(destination / 'memory.env'),
            'context_tokens': metadata['context']['tokens'], 'mode': 'continuity_only'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('backup'); export.add_argument('--binding', required=True); export.add_argument('--output', required=True)
    for name in ['verify', 'restore']:
        sub = commands.add_parser(name); sub.add_argument('--archive', required=True)
        sub.add_argument('--agent', required=True); sub.add_argument('--binary', required=True)
        if name == 'restore':
            sub.add_argument('--destination', required=True)
    args = parser.parse_args()
    try:
        if args.command == 'backup':
            result = backup(load_binding(args.binding), args.output)
        elif args.command == 'verify':
            result = verify(args.archive, args.agent, args.binary)
        else:
            result = restore(args.archive, args.agent, args.binary, args.destination)
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile, sqlite3.Error, subprocess.TimeoutExpired) as exc:
        parser.exit(1, 'Portable memory refused: ' + str(exc) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
