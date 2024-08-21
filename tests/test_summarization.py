import unittest
from models.summarization_model import summarize_text

class TestSummarization(unittest.TestCase):
    def test_summarize_text(self):
        summary = summarize_text("This is a sample text for summarization.")
        self.assertTrue(len(summary) > 0, "Summarization failed.")

if __name__ == "__main__":
    unittest.main()
