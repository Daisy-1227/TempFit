import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util.config import configure_training


class TrainingConfigTest(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "train": {"batch_size": 4, "grad_accum_every": 4},
            "distributed": {"world_size": 8, "effective_batch": 128},
        }

    def test_preserves_global_batch(self):
        original = copy.deepcopy(self.cfg)
        for world, accumulation in ((1, 32), (2, 16), (4, 8), (8, 4)):
            result = configure_training(self.cfg, world)
            self.assertEqual(result["train"]["grad_accum_every"], accumulation)
            self.assertEqual(result["distributed"]["world_size"], world)
            self.assertEqual(result["distributed"]["effective_batch"], 128)
        self.assertEqual(self.cfg, original)

    def test_explicit_accumulation_changes_global_batch(self):
        result = configure_training(self.cfg, 2, batch_size=2, grad_accum_every=2)
        self.assertEqual(result["distributed"]["effective_batch"], 8)
        self.assertEqual(result["train"], {"batch_size": 2, "grad_accum_every": 2})

    def test_resume_preserves_resolved_settings(self):
        saved = configure_training(self.cfg, 2, batch_size=2, grad_accum_every=2)
        self.assertEqual(configure_training(saved, 2), saved)

    def test_nondivisible_requires_explicit_accumulation(self):
        with self.assertRaises(ValueError):
            configure_training(self.cfg, 3)
        self.assertEqual(configure_training(self.cfg, 3, grad_accum_every=2)["distributed"]["effective_batch"], 24)

    def test_invalid_values(self):
        for args in ((0, None, None), (1, 0, None), (1, 4, 0), (1, 4, -1)):
            with self.assertRaises(ValueError):
                configure_training(self.cfg, *args)


if __name__ == "__main__":
    unittest.main()
