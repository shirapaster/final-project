import unittest
import pandas as pd
import os
from pandas import DataFrame

# Import the dataset and related variables directly from the analysis script
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_analysis import perform_welchs_anova, perform_two_way_anova

class TestDataAnalysis(unittest.TestCase):
    def setUp(self) -> None:
        """Set up a mock dataset for testing."""
        self.mock_data: DataFrame = pd.DataFrame({
            'MouseID': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6'],
            'Genotype': ['Control', 'Control', 'Ts65Dn', 'Ts65Dn', 'Control', 'Ts65Dn'],
            'Treatment': ['Saline', 'Memantine', 'Saline', 'Memantine', 'Memantine', 'Saline'],
            'BDNF_N': [0.5, 0.7, 0.8, 1.2, 0.6, 1.1],
            'pCREB_N': [0.3, 0.6, 0.4, 0.9, 0.5, 0.8]
        })

    def test_anova_bdnf(self) -> None:
        """Test Welch's ANOVA for BDNF_N levels."""
        # Perform Welch's ANOVA
        bdnf_f_stat, bdnf_p_value = perform_welchs_anova(self.mock_data, 'BDNF_N', 'Treatment')

        # Ensure F-statistic and P-value are floats
        self.assertIsInstance(bdnf_f_stat, float)
        self.assertIsInstance(bdnf_p_value, float)

        # Boundary Test: Ensure P-value is within valid range
        self.assertGreaterEqual(bdnf_p_value, 0.0)
        self.assertLessEqual(bdnf_p_value, 1.0)


    def test_two_way_anova(self) -> None:
        """Test Two-Way ANOVA for pCREB_N levels."""
        anova_results = perform_two_way_anova(self.mock_data, 'pCREB_N', 'Genotype', 'Treatment')

        # Ensure the result is a DataFrame
        self.assertIsInstance(anova_results, pd.DataFrame)

        # Check that the DataFrame contains the expected columns
        expected_columns = ['sum_sq', 'df', 'F', 'PR(>F)']
        for col in expected_columns:
            self.assertIn(col, anova_results.columns)

        # Ensure F-statistic and P-value are within valid ranges
        self.assertGreaterEqual(anova_results['F'].iloc[0], 0.0)
        self.assertGreaterEqual(anova_results['PR(>F)'].iloc[0], 0.0)
        self.assertLessEqual(anova_results['PR(>F)'].iloc[0], 1.0)

if __name__ == '__main__':
    unittest.main()
