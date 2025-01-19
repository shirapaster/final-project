import unittest
import pandas as pd
import os
from pandas import DataFrame
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Importing the functions from the data cleaning script
from data_cleaning import (
    load_data,
    drop_columns_with_many_missing,
    fill_missing_values_by_group,
    detect_outliers,
    remove_outliers,
    save_cleaned_data,
    check_group_balance
)


class TestDataCleaning(unittest.TestCase):
    def setUp(self) -> None:
        """Set up mock data for testing."""
        self.mock_data: DataFrame = pd.DataFrame({
            'MouseID': ['M1', 'M2', 'M3', 'M4'],
            'Genotype': ['A', 'B', 'A', 'B'],
            'Treatment': ['X', 'Y', 'X', 'Y'],
            'BDNF_N': [0.5, 0.7, None, 1.2],
            'pCREB_N': [0.3, 0.8, 1.1, None],
            'Extra': [None, None, None, None]  # Column with all missing values
        })
        self.test_file_path: str = './test_cleaned_data.csv'

    def tearDown(self) -> None:
        """Clean up any created files."""
        try:
            os.remove(self.test_file_path)
        except FileNotFoundError:
            pass

    def test_load_data(self) -> None:
        """Test loading a dataset (Positive Test Case)."""
        self.mock_data.to_csv(self.test_file_path, index=False)
        data: DataFrame = load_data(self.test_file_path)
        self.assertIsInstance(data, pd.DataFrame)  # Ensure the returned object is a DataFrame
        self.assertEqual(len(data), len(self.mock_data))  # Ensure all rows are loaded

    def test_drop_columns_with_many_missing(self) -> None:
        """Test dropping columns with more than 50% missing values (Edge Test Case)."""
        cleaned_data: DataFrame = drop_columns_with_many_missing(self.mock_data, threshold=0.5)
        self.assertNotIn('Extra', cleaned_data.columns)  # Ensure 'Extra' column is dropped

    def test_fill_missing_values_by_group(self) -> None:
        """Test filling missing values by group (Positive Test Case)."""
        mock_data: DataFrame = pd.DataFrame({
            'Genotype': ['A', 'A', 'B', 'B'],
            'Treatment': ['X', 'X', 'Y', 'Y'],
            'BDNF_N': [0.5, None, 0.7, None],
        })
        filled_data: DataFrame = fill_missing_values_by_group(mock_data, ['BDNF_N'], ['Genotype', 'Treatment'])
        self.assertFalse(filled_data['BDNF_N'].isnull().any())  # Ensure no missing values
        self.assertEqual(filled_data.loc[1, 'BDNF_N'], 0.5)  # Ensure correct filling with group mean

    def test_detect_outliers(self) -> None:
        """Test detecting outliers based on a flexible IQR threshold (Boundary Test Case)."""
        mock_data: DataFrame = pd.DataFrame({'BDNF_N': [1, 1.5, 2, 3, 100]})
        outliers: DataFrame = detect_outliers(mock_data, 'BDNF_N', factor=3)
        self.assertTrue(len(outliers) > 0)  # Ensure outliers are detected

    def test_remove_outliers(self) -> None:
        """Test removing outliers with a flexible threshold (Negative Test Case)."""
        mock_data: DataFrame = pd.DataFrame({'BDNF_N': [1, 1.5, 2, 3, 100]})
        cleaned_data: DataFrame = remove_outliers(mock_data, ['BDNF_N'], factor=3)
        self.assertTrue(len(cleaned_data) < len(mock_data))  # Ensure outliers are removed

    def test_check_group_balance(self) -> None:
        """Test checking group balance (Positive Test Case)."""
        mock_group_data: DataFrame = pd.DataFrame({
            'Genotype': ['Control', 'Ts65Dn', 'Control', 'Ts65Dn'],
            'Treatment': ['Saline', 'Saline', 'Memantine', 'Memantine'],
            'BDNF_N': [0.5, 0.7, 0.6, 0.8],
            'pCREB_N': [0.3, 0.4, 0.5, 0.6]
        })
        check_group_balance(mock_group_data)  # Should not raise any errors

    def test_save_cleaned_data(self) -> None:
        """Test saving cleaned data to a file (Error Test Case)."""
        save_cleaned_data(self.mock_data, self.test_file_path)
        self.assertTrue(os.path.exists(self.test_file_path))  # Ensure file is saved


if __name__ == '__main__':
    unittest.main()
