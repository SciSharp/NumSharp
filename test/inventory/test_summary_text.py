"""Guard the published summary against confusing string contract phases with replay outcomes."""

import unittest

from generate_test_inventory import summary_text


class StringSummaryTests(unittest.TestCase):
    """Keep shipped phase-2 cases visible without labeling them as unimplemented."""

    def test_phase_two_is_a_subset_not_a_deferral_count(self):
        """Check the report labels without building assemblies or scanning a large corpus.

        Returns:
            None; assertions validate the rendered Markdown.

        Raises:
            AssertionError: A phase-tag count is presented as pending work or loses its numeric value.
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
        self.assertIn("String dtype test cases (phase-2-tagged subset, not pending work) | 100 (80)", result)
        self.assertNotIn("deferred", result)
        self.assertIn("not a runtime pass rate", result)


if __name__ == "__main__":
    unittest.main()
