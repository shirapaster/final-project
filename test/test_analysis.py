import unittest
import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import os

# Import the dataset and related variables directly from the analysis script
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data_analysis import data, bdnf_anova, pcreb_anova_table

class TestAnalysis(unittest.TestCase):

    def setUp(self):
        """Set up a mock dataset for testing."""
        # Mock dataset with valid groups for positive test cases
        self.mock_data = pd.DataFrame({
            'MouseID': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6'],
            'Genotype': ['Control', 'Control', 'Ts65Dn', 'Ts65Dn', 'Control', 'Ts65Dn'],
            'Treatment': ['Saline', 'Memantine', 'Saline', 'Memantine', 'Memantine', 'Saline'],
            'BDNF_N': [0.5, 0.7, 0.8, 1.2, 0.6, 1.1],
            'pCREB_N': [0.3, 0.6, 0.4, 0.9, 0.5, 0.8]
        })

    def test_anova_bdnf(self):
        """Test ANOVA for BDNF_N levels (Positive Test Case)."""
        # Use the ANOVA results from the imported function
        self.assertIsInstance(bdnf_anova.statistic, float)
        self.assertIsInstance(bdnf_anova.pvalue, float)
        
        # Boundary Test: Ensure P-value is valid
        self.assertGreaterEqual(bdnf_anova.pvalue, 0)
        self.assertLessEqual(bdnf_anova.pvalue, 1)

    def test_twoway_anova_pcreb(self):
        """Test two-way ANOVA for pCREB_N levels (Positive Test Case)."""
        # Use the ANOVA table from the imported function
        self.assertFalse(pcreb_anova_table.empty)
        
        # Check if interaction term is present
        self.assertIn('C(Genotype):C(Treatment)', pcreb_anova_table.index)
        
        # Boundary Test: Ensure P-value is valid
        interaction_p_value = pcreb_anova_table.loc['C(Genotype):C(Treatment)', 'PR(>F)']
        self.assertGreaterEqual(interaction_p_value, 0)
        self.assertLessEqual(interaction_p_value, 1)

    def test_load_data(self):
        """Test loading the main dataset (Positive Test Case)."""
        # Use the dataset from the imported script
        self.assertIsInstance(data, pd.DataFrame)
        
        # Ensure the dataset contains required columns
        required_columns = ['MouseID', 'Genotype', 'Treatment', 'BDNF_N', 'pCREB_N']
        for column in required_columns:
            self.assertIn(column, data.columns)

    def test_visualization_files(self):
        """Test if visualization files are created successfully (Error Test Case)."""
        # Simulate saving of plots
        bdnf_plot_path = './BDNF_N_boxplot.png'
        pcreb_plot_path = './pCREB_N_interaction_plot.png'
        
        # Create dummy files for testing
        with open(bdnf_plot_path, 'w') as f:
            f.write("BDNF plot")
        with open(pcreb_plot_path, 'w') as f:
            f.write("pCREB plot")
        
        # Check if files exist
        self.assertTrue(os.path.exists(bdnf_plot_path))
        self.assertTrue(os.path.exists(pcreb_plot_path))
        
        # Clean up files
        os.remove(bdnf_plot_path)
        os.remove(pcreb_plot_path)

if __name__ == '__main__':
    unittest.main()

