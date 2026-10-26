"""Native, offline backup/restore contracts with real attributed memory."""
import hashlib
from contextlib import closing
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from test_being_inspector import fixture, BINARY

ROOT = Path(__file__).resolve().parents[1]
TOOL = Path(os.environ.get('LICORICE_PORTABLE_TEST_TOOL', ROOT / 'integrations/being-portable/memory.py')).resolve()
spec = importlib.util.spec_from_file_location('portable_memory', TOOL)
portable = importlib.util.module_from_spec(spec); spec.loader.exec_module(portable)
OPTIMIZED = Path(os.environ.get('LICORICE_PORTABLE_TEST_BINARY', ROOT / 'runs/release-20261025/staged-final-2/releases/0.2.0-alpha.1/bin/licorice-being')).resolve()


def run(command, **kwargs):
    p = subprocess.run([str(x) for x in command], capture_output=True, text=True, **kwargs)
    if p.returncode:
        raise RuntimeError(p.stdout + p.stderr)
    return json.loads(p.stdout)


class PortableTests(unittest.TestCase):
    def test_old_python_sqlite_refused_before_open(self):
        with patch.object(portable.sqlite3, 'sqlite_version_info', (3, 45, 3)), \
             patch.object(portable.sqlite3, 'connect') as connect:
            with self.assertRaisesRegex(ValueError, '>=3.53.2'):
                portable.describe(Path('/not-opened.sqlite'), 'demo', 256)
            with self.assertRaisesRegex(ValueError, '>=3.53.2'):
                portable.consistent_copy(Path('/not-opened.sqlite'), Path('/not-written.sqlite'))
            connect.assert_not_called()
    @classmethod
    def setUpClass(cls):
        cls.source = tempfile.TemporaryDirectory(prefix='licorice-portable-fixture-')
        cls.binding = fixture(cls.source.name) | {'version': 1, 'backend': 'native', 'include_episodes': True}

    @classmethod
    def tearDownClass(cls):
        cls.source.cleanup()

    def export(self, path, binding=None):
        binding = binding or self.binding
        portable.backup(binding, path)
        return path

    def altered(self, archive, output, change_metadata=None, change_database=None):
        with zipfile.ZipFile(archive) as zip:
            metadata = json.loads(zip.read('manifest.json')); database = zip.read('memory.sqlite')
        if change_metadata:
            change_metadata(metadata)
        if change_database:
            database = change_database(database)
        with zipfile.ZipFile(output, 'x') as zip:
            zip.writestr('manifest.json', json.dumps(metadata)); zip.writestr('memory.sqlite', database)
        return output

    def test_fresh_process_roundtrip_preserves_history_state_and_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); binding = root / 'source-binding.json'
            binding.write_text(json.dumps(self.binding))
            before = Path(self.binding['store']).read_bytes()
            archive = root / 'memory.licorice'
            exported = run(['python3', TOOL, 'backup', '--binding', binding, '--output', archive])
            self.assertEqual(exported['revision'], 13)
            self.assertEqual(archive.stat().st_mode & 0o777, 0o600)
            with zipfile.ZipFile(archive) as zip:
                metadata = json.loads(zip.read('manifest.json'))
                self.assertEqual(set(zip.namelist()), {'memory.sqlite', 'manifest.json'})
                self.assertNotIn(str(root), json.dumps(metadata))
                self.assertNotIn(self.binding['store'], json.dumps(metadata))
            result = run(['python3', TOOL, 'restore', '--archive', archive, '--agent', self.binding['agent_id'],
                          '--binary', BINARY, '--destination', root / 'restored'])
            state = root / 'restored'; restored = portable.load_binding(result['binding'])
            self.assertEqual(restored['relationship_id'], self.binding['relationship_id'])
            self.assertEqual(portable.sha(state / 'memory.sqlite'), metadata['database']['sha256'])
            self.assertEqual(state.stat().st_mode & 0o777, 0o700)
            self.assertTrue(all(p.stat().st_mode & 0o777 == 0o600 for p in state.iterdir()))
            for command, request in [('history', None), ('snapshot', None), ('context', {'query': portable.QUERY, 'budget_tokens': 2000, 'include_episodes': True})]:
                replies = []
                for selected in [self.binding, restored]:
                    args = [BINARY, command, '--store', selected['store'], '--agent', selected['agent_id'], '--relationship', selected['relationship_id']]
                    replies.append(run(args, input=json.dumps(request) if request else None)['result'])
                self.assertEqual(replies[0], replies[1])
            with closing(sqlite3.connect(state / 'memory.sqlite')) as db:
                # Backup is the whole identity, including another relationship.
                self.assertIsNotNone(db.execute("SELECT 1 FROM being_observations WHERE source_id='foreign-secret'").fetchone())
            self.assertEqual(before, Path(self.binding['store']).read_bytes())

    def test_committed_wal_is_included_without_copying_only_main_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); binding = fixture(root) | {'include_episodes': True}
            store = Path(binding['store'])
            with closing(sqlite3.connect(store)) as writer:
                writer.execute('PRAGMA journal_mode=WAL'); writer.execute('PRAGMA wal_autocheckpoint=0')
                writer.execute("INSERT INTO meta(key,value) VALUES('portable_wal_probe','committed')"); writer.commit()
                self.assertGreater(Path(str(store) + '-wal').stat().st_size, 0)
                before = store.read_bytes()
                archive = self.export(root / 'wal.licorice', binding)
                portable.restore(archive, binding['agent_id'], BINARY, root / 'restored')
                with closing(sqlite3.connect(root / 'restored/memory.sqlite')) as restored:
                    self.assertEqual(restored.execute("SELECT value FROM meta WHERE key='portable_wal_probe'").fetchone(), ('committed',))
                self.assertEqual(before, store.read_bytes())

    def test_wrong_owner_dimension_and_source_schema_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); archive = self.export(root / 'memory.licorice')
            with self.assertRaisesRegex(ValueError, 'owner'):
                portable.restore(archive, 'another-agent', BINARY, root / 'wrong')
            self.assertFalse((root / 'wrong').exists())
            for changed, message in [({'agent_id': 'another-agent'}, 'owner'), ({'embed_dim': 512}, 'dimension')]:
                with self.assertRaisesRegex(ValueError, message):
                    portable.backup(self.binding | changed, root / 'bad.licorice')
                self.assertFalse((root / 'bad.licorice').exists())
            copied = root / 'old.sqlite'; shutil.copyfile(self.binding['store'], copied)
            for schema in [10, 12]:
                with closing(sqlite3.connect(copied)) as db:
                    db.execute('PRAGMA user_version=' + str(schema))
                before = copied.read_bytes()
                with self.assertRaisesRegex(ValueError, 'schema11'):
                    portable.backup(self.binding | {'store': str(copied)}, root / 'bad.licorice')
                self.assertEqual(copied.read_bytes(), before)

    def test_corrupt_truncated_and_changed_contract_refused_before_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); archive = self.export(root / 'memory.licorice')
            variants = [
                self.altered(archive, root / 'changed-db', change_database=lambda body: body[:-10] + b'corruption'),
                self.altered(archive, root / 'changed-journal', change_metadata=lambda m: m.update(journal_sha256='0' * 64)),
                self.altered(archive, root / 'changed-snapshot', change_metadata=lambda m: m.update(snapshot_sha256='0' * 64)),
                self.altered(archive, root / 'changed-format', change_metadata=lambda m: m.update(version=2)),
                self.altered(archive, root / 'changed-schema', change_metadata=lambda m: m.update(schema_version=10)),
            ]
            truncated = root / 'truncated'; truncated.write_bytes(archive.read_bytes()[:100]); variants.append(truncated)
            for variant in variants:
                with self.subTest(variant=variant.name):
                    with self.assertRaises((ValueError, zipfile.BadZipFile)):
                        portable.restore(variant, self.binding['agent_id'], BINARY, root / 'refused')
                    self.assertFalse((root / 'refused').exists())

    def test_existing_files_directories_and_symlinks_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); archive = self.export(root / 'memory.licorice')
            digest = portable.sha(archive)
            with self.assertRaisesRegex(ValueError, 'exists'):
                portable.backup(self.binding, archive)
            self.assertEqual(portable.sha(archive), digest)
            existing = root / 'existing'; existing.mkdir(); sentinel = existing / 'keep'; sentinel.write_text('retained')
            empty = root / 'empty'; empty.mkdir()
            alias = root / 'alias'; alias.symlink_to(existing)
            dangling = root / 'dangling'; dangling.symlink_to(root / 'not-present')
            for target in [existing, empty, alias, dangling]:
                with self.subTest(target=target.name):
                    with self.assertRaises(ValueError):
                        portable.restore(archive, self.binding['agent_id'], BINARY, target)
            self.assertEqual(sentinel.read_text(), 'retained')
            for target in [alias, dangling]:
                with self.assertRaises(ValueError):
                    portable.backup(self.binding, target)

    def test_archive_extra_traversal_duplicate_compressed_and_symlink_members_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); archive = self.export(root / 'memory.licorice')
            with zipfile.ZipFile(archive) as zip:
                metadata = zip.read('manifest.json'); database = zip.read('memory.sqlite')
            for kind in ['extra', 'traversal', 'duplicate', 'compressed', 'symlink']:
                path = root / kind
                with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED if kind == 'compressed' else zipfile.ZIP_STORED) as zip:
                    zip.writestr('manifest.json', metadata)
                    if kind == 'symlink':
                        member = zipfile.ZipInfo('memory.sqlite'); member.external_attr = (stat.S_IFLNK | 0o777) << 16
                        zip.writestr(member, b'/etc/passwd')
                    else:
                        zip.writestr('../escape' if kind == 'traversal' else 'memory.sqlite', database)
                    if kind == 'extra':
                        zip.writestr('.env', b'private')
                    if kind == 'duplicate':
                        with self.assertWarnsRegex(UserWarning, 'Duplicate name'):
                            zip.writestr('manifest.json', metadata)
                with self.subTest(kind=kind):
                    with self.assertRaises(ValueError):
                        portable.restore(path, self.binding['agent_id'], BINARY, root / 'refused')
                    self.assertFalse((root / 'refused').exists())
            self.assertFalse((root / 'escape').exists())

    def test_interrupted_publish_never_exposes_partial_binding_and_original_survives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); archive = self.export(root / 'memory.licorice')
            original = Path(self.binding['store']).read_bytes()
            with patch.object(portable.os, 'rename', side_effect=OSError('injected prepublication failure')):
                with self.assertRaisesRegex(OSError, 'prepublication'):
                    portable.restore(archive, self.binding['agent_id'], BINARY, root / 'interrupted')
            self.assertTrue((root / 'interrupted').is_dir())
            self.assertEqual(list((root / 'interrupted').iterdir()), [])
            self.assertEqual(original, Path(self.binding['store']).read_bytes())
            self.assertTrue(portable.restore(archive, self.binding['agent_id'], BINARY, root / 'new-name')['restored'])
            self.assertTrue(portable.verify(archive, self.binding['agent_id'], BINARY)['valid'])

    def test_actual_process_death_before_and_after_atomic_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); archive = self.export(root / 'memory.licorice')
            script = '''import importlib.util,os,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('portable',sys.argv[1])
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
original=p.os.rename
def die(*args):
 if sys.argv[6]=='after':original(*args)
 os._exit(77)
p.os.rename=die
p.restore(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
'''
            for point in ['before', 'after']:
                destination = root / point
                proc = subprocess.run(['python3', '-c', script, str(TOOL), str(archive), self.binding['agent_id'], str(BINARY), str(destination), point], capture_output=True, text=True)
                self.assertEqual(proc.returncode, 77, proc.stderr)
                if point == 'before':
                    self.assertEqual(list(destination.iterdir()), [])
                else:
                    self.assertEqual({p.name for p in destination.iterdir()}, {'memory.sqlite', 'memory.env', 'binding.json', 'RESTORED.json'})
                    self.assertEqual(portable.load_binding(destination / 'binding.json')['agent_id'], self.binding['agent_id'])
                    restored = json.loads((destination / 'RESTORED.json').read_text())
                    self.assertEqual(restored['database_sha256'], portable.sha(destination / 'memory.sqlite'))
            self.assertTrue(portable.verify(archive, self.binding['agent_id'], BINARY)['valid'])

    def test_late_destination_and_backup_collisions_preserve_the_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); archive = root / 'memory.licorice'
            original_link = portable.os.link
            def late_link(source, destination):
                Path(destination).write_text('other writer won')
                return original_link(source, destination)
            with patch.object(portable.os, 'link', side_effect=late_link):
                with self.assertRaises(FileExistsError):
                    portable.backup(self.binding, archive)
            self.assertEqual(archive.read_text(), 'other writer won')
            archive.unlink(); self.export(archive)
            destination = root / 'restored'; original_unpack = portable.unpack
            def late_destination(*args):
                result = original_unpack(*args)
                destination.mkdir(); (destination / 'keep').write_text('other writer won')
                return result
            with patch.object(portable, 'unpack', side_effect=late_destination):
                with self.assertRaises(FileExistsError):
                    portable.restore(archive, self.binding['agent_id'], BINARY, destination)
            self.assertEqual((destination / 'keep').read_text(), 'other writer won')
            self.assertFalse((destination / 'binding.json').exists())

    def test_invalid_selected_runtime_is_refused_without_fallback_or_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); archive = self.export(root / 'memory.licorice')
            original = Path(self.binding['store']).read_bytes()
            with self.assertRaisesRegex(ValueError, 'return JSON'):
                portable.restore(archive, self.binding['agent_id'], '/usr/bin/true', root / 'restored')
            self.assertFalse((root / 'restored').exists())
            self.assertEqual(original, Path(self.binding['store']).read_bytes())

    def test_binding_formats_and_explicit_new_runtime_without_model_or_env_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            env_binding = {'LICORICE_BEING_BACKEND': 'native', 'LICORICE_BEING_AGENT': self.binding['agent_id'],
                           'LICORICE_BEING_BIN': str(BINARY), 'LICORICE_BEING_STORE': self.binding['store'],
                           'LICORICE_BEING_DIM': '256', 'LICORICE_BEING_RELATIONSHIP': self.binding['relationship_id'],
                           'LICORICE_BEING_EPISODES': '1', 'LICORICE_BEING_MODELS': '/missing/not-exported'}
            path = root / 'binding.json'; path.write_text(json.dumps(env_binding))
            selected = portable.load_binding(path)
            archive = self.export(root / 'memory.licorice', selected)
            result = portable.restore(archive, selected['agent_id'], OPTIMIZED, root / 'restored')
            binding = portable.load_binding(result['binding'])
            self.assertEqual(binding['binary'], str(OPTIMIZED.resolve()))
            self.assertIsNone(binding['models'])
            env = (root / 'restored/memory.env').read_text()
            self.assertIn('unset LICORICE_BEING_MODELS', env)
            self.assertNotIn('/missing/not-exported', env)
            self.assertLessEqual(result['context_tokens'], 2000)


if __name__ == '__main__':
    unittest.main()
