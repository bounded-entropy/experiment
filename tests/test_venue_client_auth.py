"""Observer progress uses the same bearer as the Desk without changing URLs."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest
from unittest.mock import AsyncMock
from unittest.mock import patch
from urllib.error import HTTPError

from rlstack.runner.venues.client import VenueClient


class ObserverAuthenticationTest(unittest.TestCase):
    def test_progress_reads_authenticated_observer(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.headers.get('Authorization') != 'Bearer observer-test-token':
                    self.send_error(401)
                    return
                body = json.dumps({'run_id': 'example', 'updates': [],
                    'extent': 'rollout', 'committed': 1, 'target': 2, 'done': False}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        with ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with patch.dict('os.environ', {'RLSTACK_HTTP_TOKEN': 'observer-test-token'}):
                    client = VenueClient(None, f'http://127.0.0.1:{server.server_port}')
                    result = client.progress('science/example')
                self.assertEqual(result['completed'], 1)
                self.assertFalse(result['done'])
            finally:
                server.shutdown()
                thread.join()


class ObserverRegistrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_follow_waits_for_exact_accepted_run_without_resubmitting(self):
        client = VenueClient(None, 'http://observer.example')
        with patch.object(client, 'progress', side_effect=[
                HTTPError('http://observer.example', 404, 'manifest pending', {}, None),
                {'completed': 1, 'planned': 1, 'extent': 'train', 'train': [], 'done': True}]) as read, \
             patch('rlstack.runner.venues.client.asyncio.sleep', new_callable=AsyncMock):
            result = await client.follow('science/exact-run', 5, every_s=.01)
        self.assertEqual(result, 'science/exact-run')
        self.assertEqual(read.call_count, 2)
        self.assertTrue(all(c.args[0] == 'science/exact-run' for c in read.call_args_list))

    async def test_follow_reports_terminal_failure(self):
        client = VenueClient(None, 'http://observer.example')
        with patch.object(client, 'progress', return_value={'status': 'failed'}):
            with self.assertRaisesRegex(RuntimeError, 'observer reports failed'):
                await client.follow('science/failed-run', 5, every_s=.01)
