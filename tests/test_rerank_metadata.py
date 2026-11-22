import unittest
from unittest.mock import patch

from app.rag.query.rerank_service import merge_rrf_and_web, rerank_documents


class RerankMetadataTest(unittest.TestCase):
    def test_disabled_model_uses_recall_scores_without_loading_reranker(self):
        state = {
            "rrf_chunks": [
                {"title": "first", "content": "a", "score": 0.9},
                {"title": "second", "content": "b", "score": 0.8},
                {"title": "third", "content": "c", "score": 0.1},
            ],
            "web_search_docs": [],
        }

        with (
            patch("app.rag.query.rerank_service.reranker_config.bge_reranker_enabled", False),
            patch("app.rag.query.rerank_service.score_and_sort_chunks") as score,
        ):
            result = rerank_documents(state)

        score.assert_not_called()
        self.assertEqual([item["title"] for item in result], ["first", "second"])

    def test_model_memory_error_falls_back_to_recall_scores(self):
        state = {
            "rrf_chunks": [
                {"title": "first", "content": "a", "score": 0.9},
                {"title": "second", "content": "b", "score": 0.8},
                {"title": "third", "content": "c", "score": 0.1},
            ],
            "web_search_docs": [],
        }

        with patch(
            "app.rag.query.rerank_service.score_and_sort_chunks",
            side_effect=OSError(1455, "页面文件太小"),
        ):
            result = rerank_documents(state)

        self.assertEqual([item["title"] for item in result], ["first", "second"])

    def test_local_source_metadata_survives_merge(self):
        local_chunk = {
            "chunk_id": 123,
            "file_title": "H3C LA2608室内无线网关 用户手册-6W100-整本手册",
            "parent_title": "## 1.2.1 配置LA2608",
            "item_name": "H3C LA2608室内无线网关",
            "title": "## 1.2.1 配置LA2608_1",
            "part": 1,
            "content": "配置正文",
            "type": "milvus",
            "score": 0.92,
        }

        merged = merge_rrf_and_web([local_chunk], [])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["chunk_id"], 123)
        self.assertEqual(merged[0]["file_title"], local_chunk["file_title"])
        self.assertEqual(merged[0]["parent_title"], local_chunk["parent_title"])
        self.assertEqual(merged[0]["item_name"], local_chunk["item_name"])
        self.assertEqual(merged[0]["part"], 1)
        self.assertEqual(merged[0]["text"], "配置正文")


if __name__ == "__main__":
    unittest.main()
