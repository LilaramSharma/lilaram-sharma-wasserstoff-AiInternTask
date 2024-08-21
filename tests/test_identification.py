import unittest
from models.identification_model import identify_objects

class TestIdentification(unittest.TestCase):
    def test_identify_objects(self):
        predictions = identify_objects('data/segmented_objects/object_0.png')
        self.assertTrue(len(predictions['labels']) > 0, "No objects were identified.")

if __name__ == "__main__":
    unittest.main()
