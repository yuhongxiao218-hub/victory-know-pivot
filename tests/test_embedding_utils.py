import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np

from app.shared.model import embedding_utils


class EmbeddingUtilsConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.original_model = embedding_utils._bge_m3_ef
        embedding_utils._bge_m3_ef = None

    def tearDown(self):
        embedding_utils._bge_m3_ef = self.original_model

    def test_bge_m3_model_is_initialized_once_across_threads(self):
        created = 0
        created_lock = threading.Lock()
        model = object()

        def fake_constructor(**_kwargs):
            nonlocal created
            with created_lock:
                created += 1
            time.sleep(0.05)
            return model

        with patch.object(embedding_utils, "BGEM3EmbeddingFunction", fake_constructor):
            with ThreadPoolExecutor(max_workers=3) as executor:
                results = list(executor.map(lambda _: embedding_utils.get_bge_m3_ef(), range(3)))

        self.assertEqual(created, 1)
        self.assertEqual(results, [model, model, model])

    def test_embedding_encoding_is_serialized(self):
        active = 0
        max_active = 0
        active_lock = threading.Lock()

        class SparseResult:
            indices = np.array([1])
            data = np.array([0.5])
            indptr = np.array([0, 1])

        class FakeModel:
            def encode_documents(self, _texts):
                nonlocal active, max_active
                with active_lock:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.05)
                with active_lock:
                    active -= 1
                return {"dense": np.array([[0.1, 0.2]]), "sparse": SparseResult()}

        with patch.object(embedding_utils, "get_bge_m3_ef", return_value=FakeModel()):
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _: embedding_utils.generate_embeddings(["text"]), range(2)))

        self.assertEqual(max_active, 1)
        self.assertEqual(results[0]["dense"], [[0.1, 0.2]])


if __name__ == "__main__":
    unittest.main()
