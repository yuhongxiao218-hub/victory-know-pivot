import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.rag.import_ import enrich_markdown_images


class FailingVisionModel:
    def __or__(self, _other):
        return self

    def invoke(self, _messages):
        raise TimeoutError("vision request timed out")


class EnrichMarkdownImagesTest(unittest.TestCase):
    def test_failed_image_description_does_not_block_import(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            image_path.write_bytes(b"image")
            image_info = [(image_path.name, image_path, ("before", "after"))]

            with patch.object(
                enrich_markdown_images.llm_provider,
                "vision_chat",
                return_value=FailingVisionModel(),
            ), patch.object(enrich_markdown_images, "load_prompt", return_value="prompt"):
                result = enrich_markdown_images.summary_image_info(image_info, "document")

        self.assertEqual(result, {})

    def test_missing_description_uses_image_alt_text(self):
        result = enrich_markdown_images.replace_old_md_content(
            "![](images/image.png)",
            {"image.png": "https://example.com/image.png"},
            {},
        )

        self.assertEqual(result, "![图片](https://example.com/image.png)")


if __name__ == "__main__":
    unittest.main()
