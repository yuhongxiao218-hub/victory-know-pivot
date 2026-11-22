import unittest
from unittest.mock import patch

from app.process.query.agent.state import create_query_default_state
from app.rag.query.item_name_confirm_service import confirm_item_name
from app.rag.query.reference_resolver_service import resolve_reference
from app.rag.query.route_response_service import build_clarification_response


class ReferenceResolverTest(unittest.TestCase):
    def test_single_recent_subject_resolves_pronoun(self):
        state = create_query_default_state(
            original_query="这台机器怎么操作？",
            history=[
                {
                    "role": "user",
                    "text": "LA2608怎么配置？",
                    "item_names": ["H3C LA2608室内无线网关"],
                }
            ],
        )

        result = resolve_reference(state)

        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.item_names, ["H3C LA2608室内无线网关"])
        self.assertEqual(result.standalone_query, "H3C LA2608室内无线网关怎么操作？")

    def test_latest_confirmed_subject_wins_in_chronological_history(self):
        state = create_query_default_state(
            original_query="它怎么操作？",
            history=[
                {"role": "user", "item_names": ["旧设备"]},
                {"role": "assistant", "item_names": []},
                {"role": "user", "item_names": ["H3C LA2608"]},
            ],
        )

        result = resolve_reference(state)

        self.assertEqual(result.item_names, ["H3C LA2608"])
        self.assertEqual(result.standalone_query, "H3C LA2608怎么操作？")

    def test_multiple_recent_subjects_require_clarification(self):
        state = create_query_default_state(
            original_query="它怎么操作？",
            history=[
                {
                    "role": "user",
                    "item_names": ["H3C LA2608", "HAK 180烫金机"],
                }
            ],
        )

        result = resolve_reference(state)

        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(len(result.item_names), 2)

    def test_missing_history_requires_clarification(self):
        state = create_query_default_state(original_query="这台机器怎么操作？")

        result = resolve_reference(state)

        self.assertEqual(result.status, "missing")
        self.assertEqual(result.item_names, [])

    def test_ambiguous_response_lists_candidates(self):
        state = create_query_default_state(
            reference_status="ambiguous",
            reference_candidates=["H3C LA2608", "HAK 180烫金机"],
        )

        answer = build_clarification_response(state)

        self.assertIn("H3C LA2608", answer)
        self.assertIn("HAK 180烫金机", answer)

    def test_explicit_model_does_not_inherit_old_subject(self):
        state = create_query_default_state(
            original_query="这个LA2608怎么操作？",
            history=[{"role": "user", "item_names": ["HAK 180烫金机"]}],
        )

        result = resolve_reference(state)

        self.assertEqual(result.status, "not_required")
        self.assertEqual(result.source, "explicit")

    def test_confirmed_history_subject_skips_llm_and_milvus(self):
        state = create_query_default_state(
            session_id="reference-test",
            original_query="它怎么操作？",
            reference_status="resolved",
            item_names=["H3C LA2608"],
            rewritten_query="H3C LA2608怎么操作？",
        )

        with (
            patch(
                "app.rag.query.item_name_confirm_service.call_llm_confirm_item_name_and_rewritten_query"
            ) as llm_call,
            patch("app.rag.query.item_name_confirm_service.milvus_hybrid_search") as milvus_call,
            patch("app.rag.query.item_name_confirm_service.save_user_chat_message") as save_message,
        ):
            result = confirm_item_name(state)

        llm_call.assert_not_called()
        milvus_call.assert_not_called()
        save_message.assert_called_once()
        self.assertEqual(result["item_names"], ["H3C LA2608"])


if __name__ == "__main__":
    unittest.main()
