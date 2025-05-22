import unittest
from endoai.core.logger import get_logger

class TestLogger(unittest.TestCase):
    def test_logger_creation(self):
        logger = get_logger("test")
        self.assertIsNotNone(logger)
        logger.info("Logger test message.")

if __name__ == "__main__":
    unittest.main()
