import unittest
from unittest.mock import patch

from app.process.query.agent.main_graph import query_graph_app
from app.process.query.agent.state import create_query_default_state
from app.rag.query.agent_trace_service import build_agent_trace
from app.rag.query.planner_schema import PlannerDecision
from app.rag.query.evidence_grader_schema import EvidenceGrade
from app.rag.query.tools.schema import SearchToolResult


class QueryGraphIntegrationTest(unittest.TestCase):
    def test_missing_reference_clarifies_before_planner(self):
        state = create_query_default_state(
            session_id="test-missing-reference",
            original_query="这台机器怎么操作？",
            is_stream=False,
        )

        with (
            patch("app.process.query.agent.nodes.node_load_context.load_query_context", return_value=[]),
            patch(
                "app.process.query.agent.nodes.node_query_planner.plan_query",
                side_effect=AssertionError("缺少历史主体时不应进入Planner"),
            ),
            patch("app.process.query.agent.nodes.node_clarify.save_routed_user_message"),
            patch("app.process.query.agent.nodes.node_answer_output.generate_answer", side_effect=lambda current: current),
        ):
            result = query_graph_app.invoke(state)

        self.assertEqual(result["reference_status"], "missing")
        self.assertEqual(result["actual_route"], "clarify")
        self.assertEqual(build_agent_trace(result)["route"], "clarify")
        self.assertIn("设备名称或型号", result["answer"])

    def test_direct_answer_skips_all_retrieval_nodes(self):
        decision = PlannerDecision(
            intent="chat",
            route="direct_answer",
            selected_tools=[],
            confidence=0.99,
            reason="普通问候",
        )
        state = create_query_default_state(
            session_id="test-direct-route",
            original_query="你好",
            is_stream=False,
        )

        with (
            patch("app.process.query.agent.nodes.node_load_context.load_query_context", return_value=[]),
            patch("app.process.query.agent.nodes.node_query_planner.plan_query", return_value=decision),
            patch("app.process.query.agent.nodes.node_direct_answer.save_routed_user_message"),
            patch("app.process.query.agent.nodes.node_answer_output.generate_answer", side_effect=lambda current: current),
        ):
            result = query_graph_app.invoke(state)

        self.assertEqual(result["planner_route"], "direct_answer")
        self.assertEqual(result["actual_route"], "direct_answer")
        self.assertEqual(build_agent_trace(result)["route"], "direct_answer")
        self.assertIn("知枢智能助手", result["answer"])
        self.assertEqual(result["embedding_chunks"], [])
        self.assertEqual(result["web_search_docs"], [])

    def test_hybrid_route_joins_local_and_web_results_before_rerank(self):
        decision = PlannerDecision(
            intent="latest_information",
            route="hybrid_rag",
            selected_tools=["local_search", "web_search"],
            confidence=0.95,
            reason="同时需要设备手册与最新外部信息",
        )
        state = create_query_default_state(
            session_id="test-hybrid-route",
            original_query="结合手册和最新信息介绍 LA2608",
            is_stream=False,
        )

        def confirm_item(current):
            current["item_names"] = ["H3C LA2608"]
            current["rewritten_query"] = current["original_query"]
            return current

        local_chunk = {
            "chunk_id": "local-1",
            "title": "LA2608 用户手册",
            "content": "本地手册内容",
            "type": "milvus",
        }
        web_doc = {"title": "最新信息", "snippet": "网页内容", "url": "https://example.com"}
        local_tool_result = SearchToolResult(
            tool_name="local_search",
            success=True,
            documents=[local_chunk],
            duration_ms=5,
        )
        web_tool_result = SearchToolResult(
            tool_name="web_search",
            success=True,
            documents=[web_doc],
            duration_ms=8,
        )

        def rerank(current):
            self.assertTrue(current["rrf_chunks"])
            self.assertEqual(current["web_search_docs"], [web_doc])
            return [{"title": "融合结果", "text": "内容", "type": "web", "score": 0.9}]

        with (
            patch("app.process.query.agent.nodes.node_load_context.load_query_context", return_value=[]),
            patch("app.process.query.agent.nodes.node_query_planner.plan_query", return_value=decision),
            patch("app.process.query.agent.nodes.node_item_name_confim.confirm_item_name", side_effect=confirm_item),
            patch(
                "app.process.query.agent.nodes.node_search_embedding.execute_search_tool",
                return_value=local_tool_result,
            ),
            patch(
                "app.process.query.agent.nodes.node_web_search_mcp.execute_search_tool",
                return_value=web_tool_result,
            ),
            patch("app.process.query.agent.nodes.node_rerank.rerank_documents", side_effect=rerank),
            patch(
                "app.process.query.agent.nodes.node_evidence_grader.grade_evidence",
                return_value=EvidenceGrade(
                    action="answer",
                    confidence=0.95,
                    reason="本地与网络证据能够支持回答",
                ),
            ),
            patch("app.process.query.agent.nodes.node_answer_output.generate_answer", side_effect=lambda current: current),
        ):
            result = query_graph_app.invoke(state)

        self.assertEqual(result["planner_route"], "hybrid_rag")
        self.assertEqual(result["actual_route"], "hybrid_rag")
        self.assertEqual(build_agent_trace(result)["route"], "hybrid_rag")
        self.assertEqual(result["reranked_docs"][0]["title"], "融合结果")
        self.assertEqual(
            {record["tool_name"] for record in result["tool_executions"]},
            {"local_search", "web_search"},
        )

    def test_empty_local_result_falls_back_to_web_once(self):
        decision = PlannerDecision(
            intent="device_knowledge",
            route="device_rag",
            selected_tools=["local_search"],
            confidence=0.9,
            reason="设备知识问题",
        )
        state = create_query_default_state(
            session_id="test-evidence-retry",
            original_query="LA2608怎么配置？",
            is_stream=False,
        )

        def confirm_item(current):
            current["item_names"] = ["H3C LA2608"]
            current["rewritten_query"] = current["original_query"]
            return current

        web_doc = {
            "title": "LA2608配置资料",
            "snippet": "通过wlan ac ip命令指定控制器地址",
            "url": "https://example.com/la2608",
        }
        local_result = SearchToolResult(
            tool_name="local_search",
            success=True,
            documents=[],
            duration_ms=3,
        )
        web_result = SearchToolResult(
            tool_name="web_search",
            success=True,
            documents=[web_doc],
            duration_ms=6,
        )

        def rerank(current):
            if not current.get("rrf_chunks") and not current.get("web_search_docs"):
                return []
            return [{"title": "LA2608配置资料", "text": "配置说明", "type": "web", "score": 0.8}]

        grades = [
            EvidenceGrade(
                action="retry",
                confidence=1.0,
                reason="本地检索无结果，使用Web兜底",
                retry_tools=["web_search"],
            ),
            EvidenceGrade(
                action="answer",
                confidence=0.95,
                reason="Web补充证据足够",
            ),
        ]

        with (
            patch("app.process.query.agent.nodes.node_load_context.load_query_context", return_value=[]),
            patch("app.process.query.agent.nodes.node_query_planner.plan_query", return_value=decision),
            patch("app.process.query.agent.nodes.node_item_name_confim.confirm_item_name", side_effect=confirm_item),
            patch(
                "app.process.query.agent.nodes.node_search_embedding.execute_search_tool",
                return_value=local_result,
            ),
            patch(
                "app.process.query.agent.nodes.node_web_search_mcp.execute_search_tool",
                return_value=web_result,
            ),
            patch("app.process.query.agent.nodes.node_rerank.rerank_documents", side_effect=rerank),
            patch(
                "app.process.query.agent.nodes.node_evidence_grader.grade_evidence",
                side_effect=grades,
            ),
            patch("app.process.query.agent.nodes.node_answer_output.generate_answer", side_effect=lambda current: current),
        ):
            result = query_graph_app.invoke(state)

        self.assertEqual(result["evidence_retry_count"], 1)
        self.assertEqual(result["evidence_action"], "answer")
        self.assertEqual(result["actual_route"], "device_rag")
        self.assertEqual(build_agent_trace(result)["route"], "device_rag")
        self.assertEqual(
            [record["tool_name"] for record in result["tool_executions"]],
            ["local_search", "web_search"],
        )

    def test_complex_plan_runs_agent_and_passes_evidence_to_answer_node(self):
        decision = PlannerDecision(
            intent="device_knowledge",
            route="device_rag",
            selected_tools=["local_search"],
            confidence=0.95,
            reason="需要先检索再比较",
            task_plan={
                "goal": "比较设备参数",
                "steps": [
                    {"id": "step_1", "description": "检索设备参数", "tool": "local_search", "success_criteria": "获得参数证据"},
                    {"id": "step_2", "description": "比较设备参数", "tool": "compare_products", "success_criteria": "形成差异结论"},
                ],
            },
        )
        state = create_query_default_state(
            session_id="test-agent-route",
            original_query="比较设备A和设备B",
            is_stream=False,
        )

        def confirm_item(current):
            current["item_names"] = ["设备A", "设备B"]
            current["rewritten_query"] = current["original_query"]
            return current

        def run_loop(current):
            current["agent_status"] = "completed"
            current["agent_finish_reason"] = "全部计划步骤完成"
            current["agent_step_count"] = 2
            current["step_observations"] = [{
                "step_id": "step_1",
                "tool_name": "local_search",
                "success": True,
                "documents": [{"title": "设备参数", "content": "设备A内存8G，设备B内存16G", "type": "milvus"}],
                "duration_ms": 2,
                "error": None,
            }]
            return current

        def generate_answer(current):
            self.assertEqual(current["reranked_docs"][0]["text"], "设备A内存8G，设备B内存16G")
            current["answer"] = "设备B内存更大。"
            return current

        with (
            patch("app.process.query.agent.nodes.node_load_context.load_query_context", return_value=[]),
            patch("app.process.query.agent.nodes.node_query_planner.plan_query", return_value=decision),
            patch("app.process.query.agent.nodes.node_item_name_confim.confirm_item_name", side_effect=confirm_item),
            patch("app.process.query.agent.nodes.node_agent_execute.run_agent_loop", side_effect=run_loop),
            patch("app.process.query.agent.nodes.node_answer_output.generate_answer", side_effect=generate_answer),
        ):
            result = query_graph_app.invoke(state)

        self.assertEqual(result["actual_route"], "agent")
        self.assertEqual(result["agent_status"], "completed")
        self.assertEqual(result["answer"], "设备B内存更大。")
        self.assertEqual(build_agent_trace(result)["plan"]["goal"], "比较设备参数")


if __name__ == "__main__":
    unittest.main()
