import unittest

from app.process.query.agent.nodes.node_evidence_response import node_evidence_response
from app.process.query.agent.state import create_query_default_state


class EvidenceResponseTest(unittest.TestCase):
    def test_refuse_explains_that_evidence_is_insufficient(self):
        state = create_query_default_state(
            session_id="test-refuse-response",
            evidence_action="refuse",
            is_stream=False,
        )
        result = node_evidence_response(state)
        self.assertIn("避免编造", result["answer"])

    def test_fail_explains_that_service_is_unavailable(self):
        state = create_query_default_state(
            session_id="test-fail-response",
            evidence_action="fail",
            is_stream=False,
        )
        result = node_evidence_response(state)
        self.assertIn("检索服务暂时不可用", result["answer"])


if __name__ == "__main__":
    unittest.main()
