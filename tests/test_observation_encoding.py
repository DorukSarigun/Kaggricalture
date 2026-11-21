import unittest

import numpy as np

from observation_encoding import encode_worker_positions


class WorkerPositionTests(unittest.TestCase):
    def test_worker_channel_keeps_raw_counts(self):
        for count in (0, 1, 2, 8, 20):
            with self.subTest(count=count):
                farm = {
                    "tiles": [[None] * 4 for _ in range(3)],
                    "farmer": [2, 1],
                    "hands": [[2, 1] for _ in range(count)],
                }
                result = encode_worker_positions(farm)
                expected = np.zeros((2, 3, 4), dtype=np.float32)
                expected[0, 1, 2] = 1.0
                expected[1, 1, 2] = count
                np.testing.assert_array_equal(result, expected)
                self.assertEqual(result.dtype, np.float32)

    def test_workers_are_counted_per_cell(self):
        farm = {
            "tiles": [[None] * 4 for _ in range(3)],
            "farmer": [0, 0],
            "hands": [[2, 1], [2, 1], [1, 2]],
        }
        result = encode_worker_positions(farm)
        expected = np.zeros((2, 3, 4), dtype=np.float32)
        expected[0, 0, 0] = 1.0
        expected[1, 1, 2] = 2.0
        expected[1, 2, 1] = 1.0
        np.testing.assert_array_equal(result, expected)
