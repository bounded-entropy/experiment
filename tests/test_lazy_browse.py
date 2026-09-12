"""Closed tree branches and unselected hosts must not read experiment data."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rlstack.data.stores.local import LocalStore
from rlstack.observe.locate import Root
from rlstack.observe.ui import ui_app
from test_ui import call


class LazyBrowseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = LocalStore(self.tmp.name)
        self.home = Path(self.tmp.name)

    def manifest(self, ref):
        path = self.home / 'runs' / ref
        path.mkdir(parents=True)
        (path / 'manifest.json').write_text('{}')
        return path

    def get(self, url, roots=None):
        status, _, body = call(ui_app(roots or [self.store]), url)
        self.assertEqual(status, '200 OK', body)
        return json.loads(body)

    def test_browse_only_scans_one_directory_and_no_result_bytes(self):
        self.manifest('family/scenario/r1')
        self.manifest('other/r2')
        import os
        original = os.scandir
        visited = []
        def scan(path):
            visited.append(Path(path))
            return original(path)
        with patch('rlstack.data.stores.local.os.scandir', side_effect=scan), \
             patch.object(self.store, '_read', side_effect=AssertionError('No result reads')), \
             patch.object(self.store, '_run_directories', side_effect=AssertionError('No full discovery')):
            data = self.get('/api/browse')
        self.assertEqual(visited, [self.home / 'runs'])
        self.assertEqual([x['name'] for x in data['entries']], ['family', 'other'])
        data = self.get('/api/browse?path=family/scenario')
        self.assertEqual(data['entries'], [dict(name='r1', kind='run', path='family/scenario/r1', folder='')])
        self.assertEqual(self.get('/api/browse?path=family/scenario/r1')['entries'], [])

    def test_pagination_preserves_exact_paths(self):
        for i in range(105): self.manifest(f'group/run{i:03d}')
        first = self.get('/api/browse?path=group')
        second = self.get('/api/browse?path=group&offset=100')
        self.assertEqual((len(first['entries']), first['next']), (100, 100))
        self.assertEqual((len(second['entries']), second['next']), (5, None))
        self.assertEqual(second['entries'][0]['path'], 'group/run100')

    def test_roots_are_explicit_and_traversal_rejected(self):
        self.manifest('family/r1')
        roots = [Root('one', self.store), Root('two', self.store)]
        with patch.object(self.store, 'run_children', side_effect=AssertionError('Closed root')):
            data = self.get('/api/browse', roots)
        self.assertEqual([r['folder'] for r in data['entries']], ['one', 'two'])
        self.assertEqual(self.get('/api/browse?root=two&path=family', roots)['entries'][0]['folder'], 'two')
        status, _, _ = call(ui_app([self.store]), '/api/browse?path=../')
        self.assertEqual(status, '400 Bad Request')

    def test_host_index_never_reads_histories_or_runs(self):
        self.store.append_host_event('chosen', {'event': 'host-up', 't': 1})
        with patch.object(self.store, '_read', side_effect=AssertionError('No histories')):
            data = self.get('/api/hosts/index')
        self.assertEqual(data['hosts'], [{'host': 'chosen', 'folder': ''}])

    def test_host_summary_and_detail_read_only_selected_host(self):
        self.store.append_host_event('chosen', {'event': 'host-up', 't': 1, 'engines': ['base']})
        self.store.append_host_event('chosen', {'event': 'attach', 't': 2, 'run_id': 'r1'})
        self.store.append_host_event('other', {'event': 'host-up', 't': 1})
        read = self.store._read
        def selected(key):
            if key.startswith('hosts/') and not key.startswith('hosts/chosen/'):
                raise AssertionError('Unselected host was read')
            if key.startswith('runs/'):
                raise AssertionError('Run was read')
            return read(key)
        with patch.object(self.store, '_read', side_effect=selected):
            summary = self.get('/api/host/chosen/summary')
            self.get('/api/host/chosen')
        self.assertEqual(summary['open_attachments'], 1)
        self.assertNotIn('metrics', summary)
        self.assertNotIn('tenancy', summary)
