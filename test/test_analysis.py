import unittest
import pandas as pd
import os
from pandas import DataFrame

# Import the dataset and related variables directly from the analysis script
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_cleaning import (
    drop_columns_with_many_missing,
    fill_missing_values_by_group,
    remove_outliers,
    check_group_balance
)
from data_analysis import perform_welchs_anova

class TestDataCleaning(unittest.TestCase):
    def setUp(self) -> None:
        """Set up a mock dataset for testing."""
        self.mock_data: DataFrame = pd.DataFrame({
            'MouseID': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6'],
            'Genotype': ['Control', 'Control', 'Ts65Dn', 'Ts65Dn', 'Control', 'Ts65Dn'],
            'Treatment': ['Saline', 'Memantine', 'Saline', 'Memantine', 'Memantine', 'Saline'],
            'BDNF_N': [0.5, 0.7, 0.8, 1.2, None, 1.1],
            'pCREB_N': [0.3, 0.6, 0.4, 0.9, 0.5, None]
        })

    def test_drop_columns_with_many_missing(self) -> None:
        """Test dropping columns with too many missing values."""
        data = self.mock_data.copy()
        cleaned_data = drop_columns_with_many_missing(data, threshold=0.5)
        self.assertNotIn('MouseID', cleaned_data.columns)
        self.assertIn('BDNF_N', cleaned_data.columns)

    def test_fill_missing_values_by_group(self) -> None:
        """Test filling missing values by group."""
        data = self.mock_data.copy()
        filled_data = fill_missing_values_by_group(data, ['BDNF_N', 'pCREB_N'], ['Genotype', 'Treatment'])
        self.assertFalse(filled_data[['BDNF_N', 'pCREB_N']].isnull().any().any())

    def test_remove_outliers(self) -> None:
        """Test removing outliers from the dataset."""
        data = self.mock_data.copy()
        cleaned_data = remove_outliers(data, ['BDNF_N', 'pCREB_N'], factor=3)
        self.assertLessEqual(len(cleaned_data), len(data))

    def test_check_group_balance(self) -> None:
        """Test group balance checking without errors."""
        data = self.mock_data.copy()
        # Ensure no exceptions raised during group balance check
        check_group_balance(data)

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
        """Test ANOVA for BDNF_N levels (Positive Test Case)."""
        bdnf_f_stat: float
        bdnf_p_value: float
        bdnf_results = perform_welchs_anova(self.mock_data, 'BDNF_N', 'Treatment')
        bdnf_f_stat = bdnf_results['F'][0]
        bdnf_p_value = bdnf_results['PR(>F)'][0]


        # Ensure F-statistic and P-value are floats
        self.assertIsInstance(bdnf_f_stat, float)
        self.assertIsInstance(bdnf_p_value, float)

        # Boundary Test: Ensure P-value is within valid range
        self.assertGreaterEqual(bdnf_p_value, 0.0)
        self.assertLessEqual(bdnf_p_value, 1.0)

if __name__ == '__main__':
    unittest.main()


