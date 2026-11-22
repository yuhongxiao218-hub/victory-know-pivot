import unittest
from unittest.mock import Mock

from app.process.query.agent.state import create_query_default_state
from app.rag.query.agent_loop_service import run_agent_loop
from app.rag.query.tools.schema import SearchToolResult


def build_state(steps: list[dict]):
    return create_query_default_state(task_plan={"goal": "完成设备资料调查", "steps": steps})


class AgentLoopServiceTest(unittest.TestCase):
    def test_continues_until_all_steps_finish(self):
        state = build_state(
            [
                {
                    "id": "step_1",
                    "description": "检索本地资料",
                    "tool": "local_search",
                    "success_criteria": "获得本地证据",
                },
                {
                    "id": "step_2",
                    "description": "检索网络资料",
                    "tool": "web_search",
                    "success_criteria": "获得网络证据",
                },
            ]
        )
        executor = Mock(
            side_effect=[
                SearchToolResult(tool_name="local_search", success=True, documents=[{"content": "本地证据"}], duration_ms=1),
                SearchToolResult(tool_name="web_search", success=True, documents=[{"snippet": "网络证据"}], duration_ms=1),
            ]
        )
        evaluator = Mock(
            return_value={
                "step_completed": True,
                "evidence_sufficient": True,
                "reason": "达到成功标准",
            }
        )

        run_agent_loop(state, executor, evaluator)

        self.assertEqual(state["agent_status"], "completed")
        self.assertEqual(state["agent_step_count"], 2)
        self.assertEqual(state["current_step_index"], 2)
        self.assertEqual(len(state["step_observations"]), 2)
        self.assertTrue(all(item["evaluation"]["evidence_sufficient"] for item in state["step_observations"]))
        self.assertEqual(len(state["tool_executions"]), 2)
        self.assertEqual([call.args[0] for call in executor.call_args_list], ["local_search", "web_search"])

    def test_insufficient_evidence_replans_then_finishes(self):
        state = build_state(
            [
                {
                    "id": "step_1",
                    "description": "检索本地资料",
                    "tool": "local_search",
                    "success_criteria": "获得完整证据",
                }
            ]
        )
        executor = Mock(
            side_effect=[
                SearchToolResult(tool_name="local_search", success=True, documents=[{"content": "证据不完整"}], duration_ms=1),
                SearchToolResult(tool_name="web_search", success=True, documents=[{"snippet": "完整证据"}], duration_ms=1),
            ]
        )
        evaluator = Mock(
            side_effect=[
                {"step_completed": True, "evidence_sufficient": False, "reason": "缺少最新资料"},
                {"step_completed": True, "evidence_sufficient": True, "reason": "证据完整"},
            ]
        )
        replanner = Mock(
            return_value={
                "action": "replan",
                "reason": "改用网络搜索补充资料",
                "revised_plan": {
                    "goal": "补充设备资料",
                    "steps": [
                        {
                            "id": "retry_1",
                            "description": "联网补充资料",
                            "tool": "web_search",
                            "success_criteria": "获得完整证据",
                        }
                    ],
                },
                "clarification": None,
            }
        )

        run_agent_loop(state, executor, evaluator, replanner)

        self.assertEqual(state["agent_status"], "completed")
        self.assertEqual(state["agent_replan_count"], 1)
        self.assertEqual(state["task_plan"]["goal"], "补充设备资料")
        self.assertEqual(len(state["step_observations"]), 2)
        self.assertEqual([call.args[0] for call in executor.call_args_list], ["local_search", "web_search"])
        replanner.assert_called_once()

    def test_insufficient_evidence_can_request_clarification(self):
        state = build_state(
            [
                {
                    "id": "step_1",
                    "description": "检索设备资料",
                    "tool": "local_search",
                    "success_criteria": "获得指定型号资料",
                }
            ]
        )
        executor = Mock(
            return_value=SearchToolResult(
                tool_name="local_search",
                success=True,
                documents=[{"content": "型号不明确"}],
                duration_ms=1,
            )
        )
        evaluator = Mock(
            return_value={
                "step_completed": False,
                "evidence_sufficient": False,
                "reason": "缺少设备型号",
            }
        )
        replanner = Mock(
            return_value={
                "action": "clarify",
                "reason": "需要明确设备",
                "revised_plan": None,
                "clarification": "请提供具体设备型号。",
            }
        )

        run_agent_loop(state, executor, evaluator, replanner)

        self.assertEqual(state["agent_status"], "needs_clarification")
        self.assertEqual(state["agent_finish_reason"], "请提供具体设备型号。")

    def test_stops_when_step_budget_is_exhausted(self):
        state = build_state(
            [
                {
                    "id": "step_1",
                    "description": "检索本地资料",
                    "tool": "local_search",
                    "success_criteria": "获得本地证据",
                },
                {
                    "id": "step_2",
                    "description": "检索网络资料",
                    "tool": "web_search",
                    "success_criteria": "获得网络证据",
                },
            ]
        )
        state["agent_max_steps"] = 1
        executor = Mock(
            return_value=SearchToolResult(
                tool_name="local_search", success=True, documents=[{"content": "证据"}], duration_ms=1
            )
        )
        evaluator = Mock(
            return_value={"step_completed": True, "evidence_sufficient": True, "reason": "完成"}
        )

        run_agent_loop(state, executor, evaluator)

        self.assertEqual(state["agent_status"], "failed")
        self.assertEqual(state["agent_finish_reason"], "执行步骤预算已用完")
        self.assertEqual(state["agent_step_count"], 1)
        self.assertEqual(len(state["step_observations"]), 1)
        self.assertEqual(executor.call_count, 1)

    def test_stops_after_tool_failure_budget_is_reached(self):
        state = build_state(
            [
                {
                    "id": "step_1",
                    "description": "检索本地资料",
                    "tool": "local_search",
                    "success_criteria": "获得本地证据",
                }
            ]
        )
        state["agent_max_tool_failures"] = 1
        executor = Mock(
            return_value=SearchToolResult(
                tool_name="local_search", success=False, documents=[], duration_ms=1, error="服务不可用"
            )
        )
        evaluator = Mock()

        run_agent_loop(state, executor, evaluator)

        self.assertEqual(state["agent_status"], "failed")
        self.assertEqual(state["agent_finish_reason"], "工具失败预算已用完")
        self.assertEqual(state["agent_tool_failure_count"], 1)
        self.assertEqual(len(state["tool_executions"]), 1)
        self.assertFalse(state["tool_executions"][0]["success"])
        self.assertEqual(executor.call_count, 1)
        evaluator.assert_not_called()

    def test_replan_budget_exhaustion_does_not_call_model(self):
        state = build_state(
            [
                {
                    "id": "step_1",
                    "description": "检索设备资料",
                    "tool": "local_search",
                    "success_criteria": "获得完整证据",
                }
            ]
        )
        state["agent_max_replans"] = 0
        executor = Mock(
            return_value=SearchToolResult(
                tool_name="local_search", success=True, documents=[{"content": "不完整"}], duration_ms=1
            )
        )
        evaluator = Mock(
            return_value={"step_completed": True, "evidence_sufficient": False, "reason": "证据不足"}
        )
        replanner = Mock()

        run_agent_loop(state, executor, evaluator, replanner)

        self.assertEqual(state["agent_status"], "needs_clarification")
        self.assertEqual(state["agent_finish_reason"], "现有证据不足，请补充设备型号或更具体的任务条件。")
        replanner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
