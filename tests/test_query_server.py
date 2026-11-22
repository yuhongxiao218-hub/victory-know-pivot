import unittest
from unittest.mock import patch

from app.api.http.query_server import invoke_query_graph


class QueryServerTest(unittest.TestCase):
    def test_sync_query_reraises_graph_error(self):
        with (
            patch("app.api.http.query_server.update_task_status"),
            patch(
                "app.api.http.query_server.query_graph_app.invoke",
                side_effect=RuntimeError("graph failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "graph failed"):
                invoke_query_graph("sync-error", "test", False)


if __name__ == "__main__":
    unittest.main()
