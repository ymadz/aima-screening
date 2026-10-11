"""Synthetic-only checks for the local WHO-labeling boundary conditions."""

from decimal import Decimal
import unittest

from src.local_who2024_cohort import normalized_hb, who_label


class Who2024CohortTests(unittest.TestCase):
    def test_strict_sex_specific_boundaries(self) -> None:
        self.assertEqual(who_label(Decimal("12.9"), "M", 40, True), 1)
        self.assertEqual(who_label(Decimal("13.0"), "M", 40, True), 0)
        self.assertEqual(who_label(Decimal("11.9"), "F", 40, True), 1)
        self.assertEqual(who_label(Decimal("12.0"), "F", 40, True), 0)

    def test_unsupported_age_or_unattested_pregnancy_rejected(self) -> None:
        with self.assertRaises(ValueError):
            who_label(Decimal("11.0"), "F", 67, True)
        with self.assertRaises(ValueError):
            who_label(Decimal("11.0"), "F", 40, False)

    def test_hb_units_and_source_value_preserved(self) -> None:
        hb, column, value, unit = normalized_hb({
            "subject_id": "synthetic", "lab_hb_g_per_l": "120",
            "lab_hb_g_per_dl": "12.0",
        })
        self.assertEqual((hb, column, value, unit),
                         (Decimal("12"), "lab_hb_g_per_l", "120", "g/L"))

    def test_mismatched_units_and_invalid_values_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalized_hb({"subject_id": "synthetic", "lab_hb_g_per_l": "120",
                           "lab_hb_g_per_dl": "13.0"})
        with self.assertRaises(ValueError):
            normalized_hb({"subject_id": "synthetic", "lab_hb_g_per_l": "",
                           "lab_hb_g_per_dl": "0"})


if __name__ == "__main__":
    unittest.main()
