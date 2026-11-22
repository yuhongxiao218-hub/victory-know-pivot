import unittest

from app.process.query.agent.state import create_query_default_state
from app.rag.query.step_evaluator_service import evaluate_latest_step


def build_state(observation: dict):
    return create_query_default_state(
        task_plan={
            "goal": "检索设备参数",
            "steps": [
                {
                    "id": "step_1",
                    "description": "检索设备网络参数",
                    "tool": "local_search",
                    "success_criteria": "证据包含网络制式和接口参数",
                }
            ],
        },
        step_observations=[observation],
    )


class StepEvaluatorServiceTest(unittest.TestCase):
    def test_tool_failure_is_incomplete_without_model_call(self):
        state = build_state(
            {
                "step_id": "step_1",
                "success": False,
                "documents": [],
                "error": "Milvus unavailable",
            }
        )

        evaluation = evaluate_latest_step(
            state,
            evaluator_invoker=lambda _: self.fail("工具失败时不应调用模型"),
        )

        self.assertFalse(evaluation.step_completed)
        self.assertFalse(evaluation.evidence_sufficient)
        self.assertEqual(state["step_evaluation"], evaluation.model_dump())
        self.assertEqual(state["step_observations"][0]["evaluation"], evaluation.model_dump())
        self.assertEqual(state["agent_status"], "evaluated")

    def test_documents_are_evaluated_against_success_criteria(self):
        state = build_state(
            {
                "step_id": "step_1",
                "success": True,
                "documents": [{"title": "参数表", "content": "支持 5G，包含千兆网口"}],
                "error": None,
            }
        )

        evaluation = evaluate_latest_step(
            state,
            evaluator_invoker=lambda _: {
                "step_completed": True,
                "evidence_sufficient": True,
                "reason": "证据覆盖网络制式和接口参数",
            },
        )

        self.assertTrue(evaluation.step_completed)
        self.assertTrue(evaluation.evidence_sufficient)
        self.assertEqual(state["agent_status"], "evaluated")
        self.assertEqual(state["step_evaluation"], evaluation.model_dump())
        self.assertEqual(state["step_observations"][0]["evaluation"], evaluation.model_dump())


if __name__ == "__main__":
    unittest.main()
