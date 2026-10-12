import unittest
from src.eyelid_information_audit import severity


class SeverityTests(unittest.TestCase):
    def test_who2024_boundaries_without_relabeling(self):
        cases=[(13,'M','non_anemia'),(12,'F','non_anemia'),(12.9,'M','mild'),
               (11.9,'F','mild'),(11,'F','mild'),(10.9,'F','moderate'),(8,'M','moderate'),(7.9,'F','severe')]
        for hb,sex,expected in cases:self.assertEqual(severity(hb,sex),expected)

    def test_unverified_measurements_rejected(self):
        for hb,sex in [(0,'M'),(float('nan'),'F'),(12,'unknown')]:
            with self.assertRaises(ValueError):severity(hb,sex)
