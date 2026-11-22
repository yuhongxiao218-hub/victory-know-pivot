import unittest

from app.process.query.agent.state import create_query_default_state
from app.rag.query.agent_trace_service import build_agent_trace
from app.api.schema.query_schema import QueryRequest, QueryResponse


class AgentTraceTest(unittest.TestCase):
    def test_debug_mode_is_opt_in(self):
        request = QueryRequest(query="你好", session_id="session-1", is_stream=True)
        self.assertFalse(request.debug)
        self.assertFalse(request.evaluation_mode)

        response = QueryResponse(
            message="ok",
            session_id="session-1",
            answer="你好",
            done_list=[],
            image_urls=[],
        )
        self.assertIsNone(response.agent_trace)

    def test_builds_lightweight_trace_from_final_state(self):
        state = create_query_default_state(
            reference_status="resolved",
            reference_source="history",
            reference_confidence=0.98,
            reference_reason="沿用最近一轮主体",
            reference_candidates=["H3C LA2608室内无线网关"],
            planner_intent="device_knowledge",
            planner_route="device_rag",
            planner_reason="需要查询设备手册",
            planner_confidence=0.91,
            tool_executions=[
                {
                    "tool_name": "local_search",
                    "execution_id": "exec-1",
                    "success": True,
                    "result_count": 2,
                    "duration_ms": 18,
                    "error": None,
                }
            ],
            evidence_retry_count=1,
            evidence_action="answer",
            evidence_confidence=0.88,
            evidence_reason="证据覆盖配置步骤",
            reranked_docs=[
                {
                    "title": "## 1.2.1 配置LA2608_1",
                    "type": "milvus",
                    "url": None,
                    "chunk_id": 123,
                    "file_title": "H3C LA2608室内无线网关 用户手册-6W100-整本手册",
                    "parent_title": "## 1.2.1 配置LA2608",
                    "item_name": "H3C LA2608室内无线网关",
                    "part": 1,
                    "text": "不应该传给前端的大段正文",
                    "content": "同样不应该进入Trace的正文",
                },
                {
                    "title": "## 1.2.1 配置LA2608_1",
                    "type": "milvus",
                    "url": None,
                    "chunk_id": 124,
                    "file_title": "H3C LA2608室内无线网关 用户手册-6W100-整本手册",
                    "parent_title": "## 1.2.1 配置LA2608",
                    "item_name": "H3C LA2608室内无线网关",
                    "part": 2,
                },
            ],
        )

        trace = build_agent_trace(state)

        self.assertEqual(trace["intent"], "device_knowledge")
        self.assertEqual(trace["route"], "device_rag")
        self.assertEqual(trace["reference"]["status"], "resolved")
        self.assertEqual(trace["tools"][0]["name"], "local_search")
        self.assertEqual(trace["retry_count"], 1)
        self.assertEqual(trace["evidence"]["action"], "answer")
        self.assertEqual(len(trace["sources"]), 2)
        self.assertEqual(trace["sources"][0]["chunk_id"], "123")
        self.assertEqual(trace["sources"][1]["chunk_id"], "124")
        self.assertEqual(trace["sources"][0]["file_title"], "H3C LA2608室内无线网关 用户手册-6W100-整本手册")
        self.assertEqual(trace["sources"][0]["parent_title"], "## 1.2.1 配置LA2608")
        self.assertEqual(trace["sources"][0]["item_name"], "H3C LA2608室内无线网关")
        self.assertEqual(trace["sources"][0]["part"], 1)
        self.assertNotIn("text", trace["sources"][0])
        self.assertNotIn("content", trace["sources"][0])
        self.assertNotIn("retrieved_contexts", trace)

        evaluation_trace = build_agent_trace(state, include_contexts=True)
        self.assertEqual(
            evaluation_trace["retrieved_contexts"],
            ["不应该传给前端的大段正文"],
        )

    def test_old_source_format_remains_supported(self):
        state = create_query_default_state(
            reranked_docs=[
                {
                    "title": "旧格式来源",
                    "type": "web",
                    "url": "https://example.com/legacy",
                    "text": "不进入Trace",
                },
                {
                    "title": "旧格式来源",
                    "type": "web",
                    "url": "https://example.com/legacy",
                },
            ]
        )

        trace = build_agent_trace(state)

        self.assertEqual(len(trace["sources"]), 1)
        self.assertEqual(trace["sources"][0]["title"], "旧格式来源")
        self.assertEqual(trace["sources"][0]["url"], "https://example.com/legacy")
        self.assertIsNone(trace["sources"][0]["chunk_id"])
        self.assertIsNone(trace["sources"][0]["file_title"])
        self.assertNotIn("text", trace["sources"][0])

    def test_web_source_still_deduplicates_by_title_and_url(self):
        state = create_query_default_state(
            reranked_docs=[
                {"title": "网页来源", "type": "web", "url": "https://example.com", "chunk_id": 1},
                {"title": "网页来源", "type": "web", "url": "https://example.com", "chunk_id": 2},
            ]
        )

        trace = build_agent_trace(state)

        self.assertEqual(len(trace["sources"]), 1)

    def test_includes_multi_step_agent_execution_trace(self):
        state = create_query_default_state(
            task_plan={
                "goal": "比较两款产品",
                "steps": [
                    {
                        "id": "step_1",
                        "description": "检索产品资料",
                        "tool": "web_search",
                        "success_criteria": "获得参数证据",
                    }
                ],
            },
            step_observations=[
                {
                    "step_id": "step_1",
                    "tool_name": "web_search",
                    "success": True,
                    "documents": [
                        {
                            "title": "产品规格页",
                            "type": "web",
                            "url": "https://example.com/spec",
                            "content": "不应进入 Trace 的正文",
                        }
                    ],
                    "duration_ms": 12,
                    "error": None,
                    "evaluation": {
                        "step_completed": True,
                        "evidence_sufficient": True,
                        "reason": "参数证据充分",
                    },
                }
            ],
            agent_step_count=1,
            agent_replan_count=1,
            agent_tool_failure_count=0,
            agent_status="completed",
            agent_finish_reason="全部步骤完成",
        )

        trace = build_agent_trace(state)

        self.assertEqual(trace["plan"]["goal"], "比较两款产品")
        self.assertEqual(trace["plan"]["steps"][0]["tool"], "web_search")
        self.assertEqual(trace["steps"][0]["result_count"], 1)
        self.assertEqual(trace["steps"][0]["results"][0]["title"], "产品规格页")
        self.assertNotIn("content", trace["steps"][0]["results"][0])
        self.assertEqual(trace["steps"][0]["evaluation"]["reason"], "参数证据充分")
        self.assertEqual(trace["agent_status"], "completed")
        self.assertEqual(trace["finish_reason"], "全部步骤完成")


if __name__ == "__main__":
    unittest.main()
