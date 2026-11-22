import unittest
from unittest.mock import patch

from app.process.query.agent.state import create_query_default_state
from app.rag.query.tools.registry import execute_search_tool


class ProductCompareToolTest(unittest.TestCase):
    def test_registered_tool_extracts_and_compares_products(self):
        state = create_query_default_state(
            task_plan={
                "goal": "比较设备 A 和设备 B 的网络能力",
                "steps": [
                    {
                        "id": "compare",
                        "description": "对比产品参数",
                        "tool": "compare_products",
                        "success_criteria": "输出参数差异和建议",
                    }
                ],
            },
            step_observations=[
                {
                    "step_id": "search_a",
                    "success": True,
                    "documents": [{"title": "设备 A", "content": "设备 A 支持 5G"}],
                },
                {
                    "step_id": "search_b",
                    "success": True,
                    "documents": [{"title": "设备 B", "content": "设备 B 仅支持 4G"}],
                },
            ],
        )
        model_output = {
            "products": [
                {"name": "设备 A", "parameters": {"网络": "5G"}},
                {"name": "设备 B", "parameters": {"网络": "4G"}},
            ],
            "differences": ["设备 A 支持 5G，设备 B 仅支持 4G"],
            "recommendation": "需要更高速网络时选择设备 A",
            "missing_information": [],
        }

        with patch(
            "app.rag.query.product_compare_service.invoke_grader_model",
            return_value=model_output,
        ):
            result = execute_search_tool("compare_products", state)

        self.assertTrue(result.success)
        self.assertEqual(result.documents[0]["type"], "action")
        self.assertEqual(result.documents[0]["comparison"], model_output)


if __name__ == "__main__":
    unittest.main()
