import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.rag.query.item_name_confirm_service import call_llm_confirm_item_name_and_rewritten_query


class ItemNameConfirmServiceTest(unittest.TestCase):
    def test_assistant_message_transcript_is_unwrapped(self):
        model = RunnableLambda(
            lambda _: AIMessage(
                content=(
                    '[{"role":"user","content":"LA2608 怎么配置？"},'
                    '{"role":"assistant","content":[{"text":"{\\"item_names\\":'
                    '[\\"LA2608\\"],\\"rewritten_query\\":\\"LA2608 怎么配置？\\"}"}]}]'
                )
            )
        )

        with patch(
            "app.rag.query.item_name_confirm_service.llm_provider.chat",
            return_value=model,
        ):
            result = call_llm_confirm_item_name_and_rewritten_query([], "LA2608 怎么配置？")

        self.assertEqual(result["item_names"], ["LA2608"])
        self.assertEqual(result["rewritten_query"], "LA2608 怎么配置？")

    def test_name_list_is_normalized_to_expected_object(self):
        model = RunnableLambda(lambda _: AIMessage(content='["运营商无线控制器"]'))

        with patch(
            "app.rag.query.item_name_confirm_service.llm_provider.chat",
            return_value=model,
        ):
            result = call_llm_confirm_item_name_and_rewritten_query(
                [],
                "运营商无线控制器示例中创建的业务 VLAN ID 是多少？",
            )

        self.assertEqual(result["item_names"], ["运营商无线控制器"])
        self.assertEqual(
            result["rewritten_query"],
            "运营商无线控制器示例中创建的业务 VLAN ID 是多少？",
        )

    def test_preserves_explicit_model_identifier_when_model_focuses_on_config_object(self):
        model = RunnableLambda(
            lambda _: AIMessage(
                content=(
                    '{"item_names":["拨号访问组"],'
                    '"rewritten_query":"配置拨号访问组 1 的 IP 许可规则"}'
                )
            )
        )

        with patch(
            "app.rag.query.item_name_confirm_service.llm_provider.chat",
            return_value=model,
        ):
            result = call_llm_confirm_item_name_and_rewritten_query(
                [],
                "给 LA2608 配置拨号访问组 1 时，允许 IP 流量的命令是什么？",
            )

        self.assertEqual(result["item_names"], ["拨号访问组", "LA2608"])

    def test_enriches_wireless_network_name_with_ssid_search_term(self):
        model = RunnableLambda(
            lambda _: AIMessage(
                content=(
                    '{"item_names":["LA2608"],'
                    '"rewritten_query":"LA2608 服务模板 1 的无线网络名称是什么？"}'
                )
            )
        )

        with patch(
            "app.rag.query.item_name_confirm_service.llm_provider.chat",
            return_value=model,
        ):
            result = call_llm_confirm_item_name_and_rewritten_query(
                [],
                "LA2608 控制器示例中，服务模板 1 配置的无线网络名称是什么？",
            )

        self.assertIn("SSID", result["rewritten_query"])
        self.assertNotIn("无线网络名称", result["rewritten_query"])


if __name__ == "__main__":
    unittest.main()
