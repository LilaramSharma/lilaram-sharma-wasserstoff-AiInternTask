import unittest
from models.text_extraction_model import extract_text_from_image

class TestTextExtraction(unittest.TestCase):
    def test_extract_text(self):
        text = extract_text_from_image('data/segmented_objects/object_0.png')
        self.assertTrue(len(text) > 0, "No text was extracted.")

if __name__ == "__main__":
    unittest.main()
