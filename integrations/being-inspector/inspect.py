#!/usr/bin/env python3
"""Build a private, offline inspector from a consistent read-only source copy."""
# This entrypoint predates Python 3.14's argparse -> dataclasses -> inspect
# import chain. Keep its directory off the CLI import path so the filename
# cannot shadow the standard library's inspect module. Assets use ROOT below.
import os as _os
import sys as _sys
if __name__ == '__main__':
    _entry_directory = _os.path.dirname(_os.path.realpath(__file__))
    _sys.path = [p for p in _sys.path
                 if _os.path.realpath(p or _os.getcwd()) != _entry_directory]
import argparse
import base64
import hashlib
import html
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parent
SCHEMA = 11
MAX_STORE_BYTES = 128 * 1024 * 1024
MAX_REPORT_BYTES = 32 * 1024 * 1024


def absolute(value, label):
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(label + ' must be absolute')
    return path.resolve()


def binding_from_file(path):
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError('Binding must be a JSON object')
    if 'LICORICE_BEING_AGENT' in raw:
        if raw.get('LICORICE_BEING_BACKEND') != 'native':
            raise ValueError('Only explicit native bindings are supported')
        return dict(binary=raw['LICORICE_BEING_BIN'], store=raw['LICORICE_BEING_STORE'],
                    agent_id=raw['LICORICE_BEING_AGENT'],
                    relationship_id=raw.get('LICORICE_BEING_RELATIONSHIP'),
                    embed_dim=int(raw.get('LICORICE_BEING_DIM', 256)))
    if raw.get('version') != 1 or raw.get('backend') != 'native':
        raise ValueError('Expected a version-1 native binding')
    return {k: raw[k] for k in ['binary', 'store', 'agent_id', 'embed_dim']} | {
        'relationship_id': raw.get('relationship_id')}


def native(binary, command, store, binding, request=None):
    args = [str(binary), command, '--store', str(store), '--agent', binding['agent_id'],
            '--embed-dim', str(binding['embed_dim'])]
    if binding.get('relationship_id'):
        args += ['--relationship', binding['relationship_id']]
    proc = subprocess.run(args, input=json.dumps(request) if request else None,
                          text=True, capture_output=True, timeout=30, check=False)
    if len(proc.stdout.encode()) > MAX_REPORT_BYTES:
        raise ValueError('Native inspection response exceeds the local report limit')
    try:
        wire = json.loads(proc.stdout)
    except (ValueError, TypeError) as exc:
        raise ValueError('Native inspection did not return JSON') from exc
    if proc.returncode or wire.get('ok') is not True:
        raise ValueError('Native inspection refused: ' + str(wire.get('error', 'unknown error')))
    return wire['result']


def visible_history(events, snapshot, relationship):
    """Filter before embedding JSON; a hidden UI row is not scope isolation."""
    claims = {c['proposal']['claim_id'] for c in snapshot['claims']}
    concepts = {c['definition']['concept_id'] for c in snapshot['concepts']}
    hypotheses = {c['definition']['set_id'] for c in snapshot['hypotheses']}
    links = {c['definition']['link_id'] for c in snapshot['links']}
    relationships = {c['definition']['relationship_id'] for c in snapshot['relationships']}
    result = []
    for event in events:
        op = event['request']['operation']
        kind, payload = op['kind'], op['payload']
        keep = False
        if kind == 'observe':
            keep = payload.get('relationship_id') in (None, relationship)
        elif kind in ('propose', 'decide', 'register_commitment', 'transition_commitment'):
            keep = payload['claim_id'] in claims
        elif kind == 'define_relationship':
            keep = payload['relationship_id'] in relationships
        elif kind == 'define_concept':
            keep = payload['concept_id'] in concepts
        elif kind == 'define_hypotheses':
            keep = payload['set_id'] in hypotheses
        elif kind in ('link_positions', 'retire_link'):
            keep = payload['link_id'] in links
        # Harness boundary/session events are not relationship-scoped evidence.
        if keep:
            result.append(event)
    return result


def capture(binding, query, budget=2000):
    if sqlite3.sqlite_version_info < (3, 53, 2):
        raise ValueError('Python SQLite must be >=3.53.2 for memory inspection; this Python uses '
                         + sqlite3.sqlite_version + '. Select an updated Python interpreter.')
    if not isinstance(budget, int) or isinstance(budget, bool) or not 64 <= budget <= 2000:
        raise ValueError('Memory allowance must be 64..2000 tokens')
    if not isinstance(query, str) or not query.strip() or len(query) > 2000:
        raise ValueError('Query must be nonempty and at most 2000 characters')
    if not isinstance(binding.get('agent_id'), str) or not binding['agent_id'].strip():
        raise ValueError('Explicit agent identity is required')
    dim = binding.get('embed_dim')
    if not isinstance(dim, int) or isinstance(dim, bool) or not 1 <= dim <= 65536:
        raise ValueError('Invalid embedding dimension')
    binary = absolute(binding['binary'], 'Binary')
    source = absolute(binding['store'], 'Store')
    if not source.is_file() or source.stat().st_size > MAX_STORE_BYTES:
        raise ValueError('Expected an existing native store of at most 128 MiB')
    if not binary.is_file():
        raise ValueError('Native binary does not exist')
    with tempfile.TemporaryDirectory(prefix='licorice-inspect-') as tmp:
        copy = Path(tmp) / 'snapshot.sqlite'
        deadline = time.monotonic() + 30
        def progress(status, remaining, total):
            if time.monotonic() > deadline or total * page_size > MAX_STORE_BYTES:
                raise ValueError('Snapshot copy exceeded the local capture bound')
        # mode=ro prevents source migrations/writes; backup includes committed WAL
        # content consistently. Never use immutable=1 or copy only the main file.
        with sqlite3.connect(source.as_uri() + '?mode=ro', uri=True, timeout=5) as src:
            src.execute('PRAGMA query_only=ON')
            src.execute('PRAGMA trusted_schema=OFF')
            page_size = src.execute('PRAGMA page_size').fetchone()[0]
            with sqlite3.connect(copy) as dst:
                src.backup(dst, pages=256, progress=progress, sleep=0.01)
        copy.chmod(0o600)
        if copy.stat().st_size > MAX_STORE_BYTES:
            raise ValueError('Snapshot copy exceeds 128 MiB')
        with sqlite3.connect(copy) as db:
            schema = db.execute('PRAGMA user_version').fetchone()[0]
            if schema != SCHEMA:
                raise ValueError('Inspector requires schema11; explicitly qualify an upgrade before inspecting')
            owner = db.execute('SELECT agent_id FROM being_owner WHERE singleton=1').fetchone()
            if owner != (binding['agent_id'],):
                raise ValueError('Store owner does not match the explicit identity')
            if db.execute('SELECT count(*) FROM being_events').fetchone()[0] > 10000:
                raise ValueError('Inspector preview supports at most 10000 journal events')
            assurance = dict(db.execute('SELECT source_id,assurance FROM being_observations'))
        snapshot = native(binary, 'snapshot', copy, binding)
        history = native(binary, 'history', copy, binding)
        context = native(binary, 'context', copy, binding, {
            'query': query, 'budget_tokens': budget, 'include_episodes': True})
        revision = snapshot['revision']
        if snapshot['agent_id'] != binding['agent_id'] or context['continuity']['revision'] != revision:
            raise ValueError('Native results disagree about identity/revision')
        if (history[-1]['revision'] if history else 0) != revision:
            raise ValueError('Journal and snapshot revisions disagree')
        usage = context['usage']
        if usage['budget_tokens'] != budget or not 0 <= usage['tokens'] <= budget:
            raise ValueError('Native packet violated the selected context allowance')
        if usage['tokenizer'] != 'cl100k_base:ordinary' or context['evidence_mode'] != 'continuity_only':
            raise ValueError('Unexpected native counting/evidence mode')
        history = visible_history(history, snapshot, binding.get('relationship_id'))
        observations = [{
            'source_id': e['request']['request_id'],
            'observation': e['request']['operation']['payload'],
            'assurance': assurance[e['request']['request_id']],
            'recorded_revision': e['revision'],
        } for e in history if e['request']['operation']['kind'] == 'observe']
        return {
            'format': 'licorice-inspection-v1', 'agent_id': binding['agent_id'],
            'relationship_id': binding.get('relationship_id'), 'revision': revision,
            'query': query, 'snapshot': snapshot, 'history': history,
            'observations': observations, 'context': context,
            'limits': {
                'source_copy': 'SQLite consistent backup from a read-only source connection',
                'scope': 'Global records plus the selected relationship; not a remote authorization boundary',
                'context': 'Continuity-only preview; optional factual embedding retrieval is not executed',
                'episodes': 'Native context considers at most 128 scoped episode candidates; this report retains all scoped observations within the inspector limit',
                'history': 'Relationship-visible events; session and boundary metadata omitted',
                'privacy': 'Private memory export. Do not place in a public download directory',
            },
        }


def render(report):
    body = json.dumps(report, ensure_ascii=False, separators=(',', ':')).encode()
    if len(body) > MAX_REPORT_BYTES:
        raise ValueError('Inspection exceeds 32 MiB; no truncated report is produced')
    payload = base64.b64encode(body).decode('ascii')
    script = (ROOT / 'viewer.js').read_text().replace('__LICORICE_DATA_BASE64__', payload)
    style = (ROOT / 'viewer.css').read_text()
    digest = lambda s: base64.b64encode(hashlib.sha256(s.encode()).digest()).decode()
    policy = ("default-src 'none'; script-src 'sha256-" + digest(script) +
              "'; style-src 'sha256-" + digest(style) +
              "'; connect-src 'none'; img-src 'none'; base-uri 'none'; form-action 'none'")
    template = (ROOT / 'viewer.html').read_text()
    return (template.replace('__CSP__', html.escape(policy, quote=True))
            .replace('__STYLE__', style).replace('__SCRIPT__', script))


def write_private(path, text):
    path = output_path(path)
    # O_EXCL and O_NOFOLLOW refuse a leaf symlink even if it appears after
    # preflight. Resolve the parent only; never follow the output leaf.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def output_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError('Output must be absolute')
    path = path.parent.resolve() / path.name
    if path.is_symlink():
        raise ValueError('Output must not be a symlink')
    if not path.parent.is_dir():
        raise ValueError('Output parent must be an existing private directory')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', type=Path, help='Explicit native v1 or setup.py environment JSON binding')
    parser.add_argument('--binary')
    parser.add_argument('--store')
    parser.add_argument('--agent')
    parser.add_argument('--relationship')
    parser.add_argument('--embed-dim', type=int, default=None)
    parser.add_argument('--query', default='current commitments, methods and changed judgments')
    parser.add_argument('--budget-tokens', type=int, default=2000)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        explicit = [args.binary, args.store, args.agent, args.relationship, args.embed_dim]
        if args.binding:
            if any(v is not None for v in explicit):
                raise ValueError('Use the complete binding or explicit flags; do not mix identities')
            binding = binding_from_file(args.binding)
        else:
            if not all([args.binary, args.store, args.agent]):
                raise ValueError('Supply --binding or all of --binary, --store and --agent')
            binding = dict(binary=args.binary, store=args.store, agent_id=args.agent,
                           relationship_id=args.relationship, embed_dim=256 if args.embed_dim is None else args.embed_dim)
        output = output_path(args.output)
        if output.exists():
            raise ValueError('Output already exists; use a new private filename')
        report = capture(binding, args.query, args.budget_tokens)
        write_private(output, render(report))
        print(json.dumps({'output': str(output), 'agent': report['agent_id'],
                          'revision': report['revision'], 'tokens': report['context']['usage']['tokens'],
                          'mode': 'private_offline_continuity_inspection'}))
    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error, subprocess.TimeoutExpired) as exc:
        parser.exit(1, 'Inspection refused: ' + str(exc) + '\n')


if __name__ == '__main__':
    main()
