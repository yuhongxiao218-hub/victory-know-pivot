import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.http.query_server import app, query_graph_app
from app.process.query.agent.state import create_query_default_state


class AgentWebIntegrationTest(unittest.TestCase):
    def test_debug_query_returns_agent_plan_and_execution_trace(self):
        result = create_query_default_state(
            session_id="web-agent-test",
            original_query="比较设备A和设备B",
            answer="设备B内存更大。",
            image_urls=[],
            actual_route="agent",
            task_plan={
                "goal": "比较设备参数",
                "steps": [
                    {"id": "step_1", "description": "检索参数", "tool": "local_search", "success_criteria": "获得参数"},
                    {"id": "step_2", "description": "比较参数", "tool": "compare_products", "success_criteria": "形成结论"},
                ],
            },
            agent_status="completed",
            agent_finish_reason="全部计划步骤均已完成且证据充分",
            agent_step_count=2,
        )

        with patch.object(query_graph_app, "invoke", return_value=result):
            response = TestClient(app).post(
                "/query",
                json={"query": "比较设备A和设备B", "session_id": "web-agent-test", "is_stream": False, "debug": True},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answer"], "设备B内存更大。")
        self.assertEqual(payload["agent_trace"]["route"], "agent")
        self.assertEqual(payload["agent_trace"]["plan"]["goal"], "比较设备参数")
        self.assertEqual(payload["agent_trace"]["agent_status"], "completed")

    def test_debug_query_exposes_replan_clarification_and_budget_states(self):
        cases = [
            ("completed", "重规划后完成", 1),
            ("needs_clarification", "请补充具体设备型号。", 0),
            ("failed", "执行步骤预算已用完", 0),
        ]
        client = TestClient(app)

        for index, (status, reason, replan_count) in enumerate(cases):
            with self.subTest(status=status):
                session_id = f"web-agent-state-{index}"
                result = create_query_default_state(
                    session_id=session_id,
                    original_query="执行复杂任务",
                    answer=reason,
                    image_urls=[],
                    actual_route="agent",
                    agent_status=status,
                    agent_finish_reason=reason,
                    agent_replan_count=replan_count,
                )
                with patch.object(query_graph_app, "invoke", return_value=result):
                    response = client.post(
                        "/query",
                        json={"query": "执行复杂任务", "session_id": session_id, "is_stream": False, "debug": True},
                    )

                trace = response.json()["agent_trace"]
                self.assertEqual(trace["agent_status"], status)
                self.assertEqual(trace["finish_reason"], reason)
                self.assertEqual(trace["replan_count"], replan_count)


if __name__ == "__main__":
    unittest.main()
