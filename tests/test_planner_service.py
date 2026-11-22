import unittest

from app.process.query.agent.nodes.node_query_planner import node_query_planner
from app.process.query.agent.state import create_query_default_state
from app.rag.query.planner_service import plan_query


class PlannerServiceTest(unittest.TestCase):
    def test_valid_web_decision(self):
        state = create_query_default_state(
            session_id="planner_web_test",
            original_query="H3C最近发布了什么设备？",
        )

        decision = plan_query(
            state,
            planner_invoker=lambda _: {
                "intent": "latest_information",
                "route": "web_rag",
                "selected_tools": ["web_search"],
                "confidence": 0.96,
                "reason": "问题具有明显时效性",
            },
        )

        self.assertEqual(decision.route, "web_rag")
        self.assertEqual(decision.selected_tools, ["web_search"])

    def test_invalid_decision_falls_back_to_device_rag(self):
        state = create_query_default_state(
            session_id="planner_fallback_test",
            original_query="LA2608怎么配置？",
        )

        decision = plan_query(
            state,
            planner_invoker=lambda _: {
                "intent": "device_knowledge",
                "route": "web_rag",
                "selected_tools": ["local_search"],
                "confidence": 0.8,
                "reason": "故意构造的冲突结果",
            },
        )

        self.assertEqual(decision.route, "device_rag")
        self.assertEqual(decision.selected_tools, ["local_search"])
        self.assertEqual(decision.confidence, 0.0)

    def test_single_object_list_is_unwrapped(self):
        state = create_query_default_state(
            session_id="planner_wrapped_test",
            original_query="LA2608怎么配置？",
        )

        decision = plan_query(
            state,
            planner_invoker=lambda _: [{
                "intent": "device_knowledge",
                "route": "device_rag",
                "selected_tools": ["local_search"],
                "confidence": 0.9,
                "reason": "设备配置问题",
            }],
        )

        self.assertEqual(decision.route, "device_rag")
        self.assertEqual(decision.confidence, 0.9)

    def test_conflicting_intent_and_route_falls_back(self):
        state = create_query_default_state(
            session_id="planner_conflict_test",
            original_query="你好",
        )

        decision = plan_query(
            state,
            planner_invoker=lambda _: {
                "intent": "chat",
                "route": "web_rag",
                "selected_tools": ["web_search"],
                "confidence": 0.99,
                "reason": "故意构造的意图与路由冲突",
            },
        )

        self.assertEqual(decision.route, "device_rag")
        self.assertEqual(decision.confidence, 0.0)

    def test_empty_question_routes_to_clarify(self):
        state = create_query_default_state(
            session_id="planner_empty_test",
            original_query="   ",
        )

        decision = plan_query(state, planner_invoker=lambda _: self.fail("不应调用模型"))

        self.assertEqual(decision.route, "clarify")
        self.assertEqual(decision.selected_tools, [])

    def test_node_writes_decision_to_state(self):
        state = create_query_default_state(
            session_id="planner_node_test",
            original_query="你好",
            is_stream=False,
        )

        result = node_query_planner(
            state,
            planner_invoker=lambda _: {
                "intent": "chat",
                "route": "direct_answer",
                "selected_tools": [],
                "confidence": 0.99,
                "reason": "普通问候",
            },
        )

        self.assertEqual(result["planner_intent"], "chat")
        self.assertEqual(result["planner_route"], "direct_answer")
        self.assertEqual(result["selected_tools"], [])

    def test_complex_request_writes_multi_step_plan(self):
        state = create_query_default_state(
            session_id="planner_agent_test",
            original_query="比较两款设备的本地参数和最新价格",
            is_stream=False,
        )
        raw_decision = {
            "intent": "latest_information",
            "route": "hybrid_rag",
            "selected_tools": ["local_search", "web_search"],
            "confidence": 0.94,
            "reason": "需要收集两类证据后进行对比",
            "task_plan": {
                "goal": "比较两款设备",
                "steps": [
                    {"id": "step_1", "description": "检索本地参数", "tool": "local_search", "success_criteria": "获得设备参数"},
                    {"id": "step_2", "description": "检索最新价格", "tool": "web_search", "success_criteria": "获得近期价格"},
                    {"id": "step_3", "description": "对比产品", "tool": "compare_products", "success_criteria": "形成参数差异和建议"},
                ],
            },
        }

        result = node_query_planner(state, planner_invoker=lambda _: raw_decision)

        self.assertEqual(result["task_plan"]["goal"], "比较两款设备")
        self.assertEqual(len(result["task_plan"]["steps"]), 3)


if __name__ == "__main__":
    unittest.main()
