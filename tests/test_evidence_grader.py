import unittest

from pydantic import ValidationError

from app.process.query.agent.state import create_query_default_state
from app.rag.query.evidence_grader_schema import EvidenceGrade
from app.rag.query.evidence_grader_service import grade_evidence


class EvidenceGraderTest(unittest.TestCase):
    def test_retry_action_requires_retry_tools(self):
        with self.assertRaises(ValidationError):
            EvidenceGrade(
                action="retry",
                confidence=0.8,
                reason="需要补充检索",
            )

    def test_no_evidence_retries_untried_eligible_tool(self):
        state = create_query_default_state(
            item_names=["H3C LA2608"],
            tool_executions=[
                {
                    "tool_name": "local_search",
                    "success": True,
                    "result_count": 0,
                    "duration_ms": 5,
                    "error": None,
                }
            ],
        )

        grade = grade_evidence(state)

        self.assertEqual(grade.action, "retry")
        self.assertEqual(grade.retry_tools, ["web_search"])

    def test_no_evidence_after_retry_refuses_when_tools_completed(self):
        state = create_query_default_state(
            item_names=["H3C LA2608"],
            evidence_retry_count=1,
            tool_executions=[
                {
                    "tool_name": "local_search",
                    "success": True,
                    "result_count": 0,
                    "duration_ms": 5,
                    "error": None,
                }
            ],
        )

        grade = grade_evidence(state)

        self.assertEqual(grade.action, "refuse")

    def test_local_insufficiency_overrides_model_retry_choice_with_web(self):
        state = create_query_default_state(
            item_names=["H3C LA2608"],
            reranked_docs=[
                {
                    "title": "LA2608概览",
                    "text": "设备支持蜂窝网络",
                    "type": "milvus",
                    "score": 0.4,
                }
            ],
            tool_executions=[
                {
                    "tool_name": "local_search",
                    "success": True,
                    "result_count": 1,
                    "duration_ms": 5,
                    "error": None,
                }
            ],
        )

        grade = grade_evidence(
            state,
            grader_invoker=lambda _: {
                "action": "retry",
                "confidence": 0.9,
                "reason": "需要补充资料",
                "retry_tools": ["hyde_search"],
            },
        )

        self.assertEqual(grade.action, "retry")
        self.assertEqual(grade.retry_tools, ["web_search"])

    def test_all_tools_failed_after_retry_returns_fail(self):
        state = create_query_default_state(
            item_names=["H3C LA2608"],
            evidence_retry_count=1,
            tool_executions=[
                {
                    "tool_name": "local_search",
                    "success": False,
                    "result_count": 0,
                    "duration_ms": 5,
                    "error": "Milvus unavailable",
                },
                {
                    "tool_name": "hyde_search",
                    "success": False,
                    "result_count": 0,
                    "duration_ms": 5,
                    "error": "Embedding unavailable",
                },
            ],
        )

        grade = grade_evidence(state)

        self.assertEqual(grade.action, "fail")

    def test_documents_are_graded_by_structured_model_output(self):
        state = create_query_default_state(
            original_query="LA2608怎么配置？",
            reranked_docs=[
                {
                    "title": "LA2608 用户手册",
                    "text": "通过wlan ac ip命令指定控制器地址",
                    "type": "milvus",
                    "score": 0.9,
                }
            ],
        )

        grade = grade_evidence(
            state,
            grader_invoker=lambda _: {
                "action": "answer",
                "confidence": 0.96,
                "reason": "证据直接包含配置命令",
                "retry_tools": [],
            },
        )

        self.assertEqual(grade.action, "answer")
        self.assertEqual(grade.confidence, 0.96)


if __name__ == "__main__":
    unittest.main()
