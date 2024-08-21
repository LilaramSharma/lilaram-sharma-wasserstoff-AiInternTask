import unittest
from models.segmentation_model import segment_image

class TestSegmentation(unittest.TestCase):
    def test_segment_image(self):
        predictions = segment_image('data/input_images/sample_image.jpg')
        self.assertTrue(len(predictions['boxes']) > 0, "No objects were detected.")

if __name__ == "__main__":
    unittest.main()
