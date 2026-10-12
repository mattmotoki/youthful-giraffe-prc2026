"""Controls for the released lineage AWS-profile fact."""
import unittest

from lineage_profile import profile_is_configured


class ProfileConfiguredTests(unittest.TestCase):
    def test_absent_profile_is_false(self):
        self.assertFalse(profile_is_configured(None, "[fake-profile]\n"))

    def test_empty_profile_is_false(self):
        self.assertFalse(profile_is_configured("", "[fake-profile]\n"))

    def test_present_missing_profile_is_false(self):
        self.assertFalse(profile_is_configured("fake-selected", "[fake-other]\n"))

    def test_present_configured_profile_is_true(self):
        self.assertTrue(profile_is_configured("fake-selected", "[fake-selected]\n"))

    def test_longer_profile_does_not_match(self):
        self.assertFalse(profile_is_configured("fake-selected", "[fake-selected-longer]\n"))

    def test_exact_profile_among_sections_matches(self):
        text = "[fake-other]\nkey=x\n [fake-selected] \nkey=y\n"
        self.assertTrue(profile_is_configured("fake-selected", text))


if __name__ == "__main__":
    unittest.main()
