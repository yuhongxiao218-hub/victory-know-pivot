import unittest
from unittest.mock import Mock, patch

from app.process.query.agent.state import create_query_default_state
from app.process.query.agent.state import merge_tool_executions
from app.rag.query.tools.registry import (
    SEARCH_TOOL_REGISTRY,
    SearchToolDefinition,
    execute_search_tool,
    list_search_tools,
)


class SearchToolTest(unittest.TestCase):
    def test_registry_exposes_stable_tool_metadata(self):
        tools = {tool["name"]: tool for tool in list_search_tools()}
        self.assertEqual(
            set(tools),
            {"local_search", "hyde_search", "web_search", "compare_products"},
        )
        self.assertTrue(tools["local_search"]["requires_item_name"])
        self.assertFalse(tools["web_search"]["requires_item_name"])

    def test_successful_tool_returns_uniform_result(self):
        handler = Mock(return_value=[{"chunk_id": "1"}])
        definition = SearchToolDefinition(
            name="local_search",
            description="测试工具",
            requires_item_name=True,
            handler=handler,
        )
        state = create_query_default_state(item_names=["H3C LA2608"])

        with patch.dict(SEARCH_TOOL_REGISTRY, {"local_search": definition}):
            result = execute_search_tool("local_search", state)

        self.assertTrue(result.success)
        self.assertEqual(result.documents, [{"chunk_id": "1"}])
        self.assertEqual(result.to_trace()["result_count"], 1)

    def test_required_item_name_is_checked_before_handler(self):
        handler = Mock()
        definition = SearchToolDefinition(
            name="local_search",
            description="测试工具",
            requires_item_name=True,
            handler=handler,
        )
        state = create_query_default_state(item_names=[])

        with patch.dict(SEARCH_TOOL_REGISTRY, {"local_search": definition}):
            result = execute_search_tool("local_search", state)

        self.assertFalse(result.success)
        self.assertIn("item_names", result.error)
        handler.assert_not_called()

    def test_handler_exception_becomes_structured_failure(self):
        handler = Mock(side_effect=ConnectionError("Milvus connection refused"))
        definition = SearchToolDefinition(
            name="local_search",
            description="测试工具",
            requires_item_name=True,
            handler=handler,
        )
        state = create_query_default_state(item_names=["H3C LA2608"])

        with patch.dict(SEARCH_TOOL_REGISTRY, {"local_search": definition}):
            result = execute_search_tool("local_search", state)

        self.assertFalse(result.success)
        self.assertEqual(result.documents, [])
        self.assertIn("Milvus", result.error)

    def test_tool_execution_reducer_deduplicates_same_execution(self):
        record = {
            "execution_id": "same-execution",
            "tool_name": "local_search",
            "success": True,
        }
        merged = merge_tool_executions([record], [record])
        self.assertEqual(merged, [record])


if __name__ == "__main__":
    unittest.main()
