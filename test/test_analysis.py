import unittest
import pandas as pd
import os
from pandas import DataFrame
from typing import Tuple

# Import the dataset and related variables directly from the analysis script
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_analysis import perform_anova

class TestAnalysis(unittest.TestCase):
    def setUp(self) -> None:
        """Set up a mock dataset for testing."""
        # Mock dataset with valid groups for positive test cases
        self.mock_data: DataFrame = pd.DataFrame({
            'MouseID': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6'],
            'Genotype': ['Control', 'Control', 'Ts65Dn', 'Ts65Dn', 'Control', 'Ts65Dn'],
            'Treatment': ['Saline', 'Memantine', 'Saline', 'Memantine', 'Memantine', 'Saline'],
            'BDNF_N': [0.5, 0.7, 0.8, 1.2, 0.6, 1.1],
            'pCREB_N': [0.3, 0.6, 0.4, 0.9, 0.5, 0.8]
        })

    def test_anova_bdnf(self) -> None:
        """Test ANOVA for BDNF_N levels (Positive Test Case)."""
        bdnf_f_stat, bdnf_p_value = perform_anova(self.mock_data, 'BDNF_N', 'Treatment')
        
        # Ensure F-statistic and P-value are floats
        self.assertIsInstance(bdnf_f_stat, float)
        self.assertIsInstance(bdnf_p_value, float)

        # Boundary Test: Ensure P-value is within valid range
        self.assertGreaterEqual(bdnf_p_value, 0)
        self.assertLessEqual(bdnf_p_value, 1)

    def test_anova_pcreb(self) -> None:
        """Test ANOVA for pCREB_N levels (Positive Test Case)."""
        pcreb_f_stat, pcreb_p_value = perform_anova(self.mock_data, 'pCREB_N', 'Treatment')
        
        # Ensure F-statistic and P-value are floats
        self.assertIsInstance(pcreb_f_stat, float)
        self.assertIsInstance(pcreb_p_value, float)

        # Boundary Test: Ensure P-value is within valid range
        self.assertGreaterEqual(pcreb_p_value, 0)
        self.assertLessEqual(pcreb_p_value, 1)

if __name__ == '__main__':
    unittest.main()
