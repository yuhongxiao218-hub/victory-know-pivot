import unittest
from unittest.mock import Mock

from app.process.query.agent.state import create_query_default_state
from app.rag.query.plan_executor_service import AgentExecutionLimitError, execute_next_plan_step
from app.rag.query.tools.schema import SearchToolResult


class PlanExecutorServiceTest(unittest.TestCase):
    def setUp(self):
        self.state = create_query_default_state(
            task_plan={
                "goal": "依次检索本地和网络资料",
                "steps": [
                    {
                        "id": "step_1",
                        "description": "检索本地资料",
                        "tool": "local_search",
                        "success_criteria": "获得本地文档",
                    },
                    {
                        "id": "step_2",
                        "description": "检索网络资料",
                        "tool": "web_search",
                        "success_criteria": "获得网页文档",
                    },
                ],
            }
        )

    def test_executes_registered_tools_in_plan_order(self):
        executor = Mock(
            side_effect=[
                SearchToolResult(
                    tool_name="local_search",
                    execution_id="execution-1",
                    success=True,
                    documents=[{"chunk_id": "1"}],
                    duration_ms=10,
                ),
                SearchToolResult(
                    tool_name="web_search",
                    execution_id="execution-2",
                    success=True,
                    documents=[{"url": "https://example.com"}],
                    duration_ms=20,
                ),
            ]
        )

        execute_next_plan_step(self.state, executor)
        execute_next_plan_step(self.state, executor)

        self.assertEqual(
            [call.args[0] for call in executor.call_args_list],
            ["local_search", "web_search"],
        )
        self.assertEqual(self.state["current_step_index"], 2)
        self.assertEqual(self.state["agent_step_count"], 2)
        self.assertEqual(len(self.state["step_observations"]), 2)
        self.assertEqual(len(self.state["tool_executions"]), 2)

    def test_failed_step_is_recorded_without_advancing(self):
        executor = Mock(
            return_value=SearchToolResult(
                tool_name="local_search",
                execution_id="execution-failed",
                success=False,
                documents=[],
                duration_ms=5,
                error="Milvus unavailable",
            )
        )

        execute_next_plan_step(self.state, executor)

        self.assertEqual(self.state["current_step_index"], 0)
        self.assertEqual(self.state["agent_status"], "awaiting_evaluation")
        self.assertEqual(self.state["agent_tool_failure_count"], 1)
        self.assertEqual(self.state["step_observations"][0]["error"], "Milvus unavailable")

    def test_rejects_a_third_identical_action(self):
        executor = Mock(
            return_value=SearchToolResult(
                tool_name="local_search",
                success=False,
                documents=[],
                duration_ms=1,
                error="temporary failure",
            )
        )
        self.state["agent_max_tool_failures"] = 3

        execute_next_plan_step(self.state, executor)
        execute_next_plan_step(self.state, executor)

        with self.assertRaisesRegex(AgentExecutionLimitError, "重复调用预算"):
            execute_next_plan_step(self.state, executor)

        self.assertEqual(executor.call_count, 2)


if __name__ == "__main__":
    unittest.main()
