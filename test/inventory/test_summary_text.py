"""Guard the published summary against confusing string contract phases with replay outcomes."""

import unittest

from generate_test_inventory import summary_text


class StringSummaryTests(unittest.TestCase):
    """Keep shipped phase-2 cases visible without labeling them as unimplemented."""

    def test_phase_two_is_a_subset_not_a_deferral_count(self):
        """Keep the public summary about contracts, not the implementation stages that produced them.

        Returns:
            None; assertions validate the rendered Markdown.

        Raises:
            AssertionError: A generator phase leaks into the public label, or the total count is lost.
        """
        # A tiny synthetic report isolates presentation from the changing repository-wide inventory totals.
        tests = dict.fromkeys((
            "methods", "dataRows", "dataRowMethods", "dynamicMethods", "oracleMethods",
            "oracleTestClasses", "fuzzMatrixMethods",
        ), 0)
        tests.update(statuses={}, projects={})
        oracle = dict.fromkeys((
            "corpusRows", "specializedOracleCases", "totalSerializedContracts", "opKeys",
            "hostPinRecords", "thinOpLabels", "singleLayoutOpLabels", "singleDtypeOpLabels",
            "opLabelsWithErrors", "errorRows",
        ), 0)
        oracle.update(stringOracleCases=100, stringOraclePhase2Cases=80)
        result = summary_text({"summary": {"tests": tests, "oracle": oracle}})
        self.assertIn("| String dtype test cases | 100 |", result)
        self.assertNotIn("phase", result)
        self.assertNotIn("deferred", result)
        self.assertIn("not a runtime pass rate", result)


if __name__ == "__main__":
    unittest.main()
