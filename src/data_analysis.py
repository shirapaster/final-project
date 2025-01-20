import pandas as pd
from pandas import DataFrame
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene
from statsmodels.stats.anova import anova_lm
import seaborn as sns
from typing import Tuple

# Set style for visualizations
sns.set(style="whitegrid", palette="muted")

def load_data(file_path: str) -> DataFrame:
    """Load the dataset from the specified file path."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        raise
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading the file: {e}")
        raise

def check_normality(data: pd.DataFrame, column: str, group_col: str) -> None:
    """
    Check normality of data within groups using the Shapiro-Wilk test.

    Args:
        data (pd.DataFrame): Data to analyze.
        column (str): Column to test for normality.
        group_col (str): Grouping column.
    """
    print("\n--- Checking Normality ---")
    for name, group in data.groupby(group_col):
        stat, p_value = shapiro(group[column].dropna())
        print(f"Group: {name}, W={stat:.4f}, p={p_value:.4f}")
        if p_value < 0.05:
            print(f"  --> Group '{name}' does NOT follow a normal distribution (p < 0.05).")
        else:
            print(f"  --> Group '{name}' follows a normal distribution (p >= 0.05).")

def check_homogeneity(data: pd.DataFrame, column: str, group_col: str) -> None:
    """
    Check homogeneity of variances using Levene's test.

    Args:
        data (pd.DataFrame): Data to analyze.
        column (str): Column to test.
        group_col (str): Grouping column.
    """
    groups = [group[column].dropna() for _, group in data.groupby(group_col)]
    stat, p_value = levene(*groups)
    print(f"\nLevene's Test: W={stat:.4f}, p={p_value:.4f}")
    if p_value < 0.05:
        print("  --> Variances are NOT equal (p < 0.05). Consider using Welch's ANOVA.")
    else:
        print("  --> Variances are equal (p >= 0.05).")

def perform_welchs_anova(data: pd.DataFrame, dependent_var: str, group_var: str) -> Tuple[float, float]:
    """
    Perform Welch's ANOVA on the data.

    Returns:
        Tuple[float, float]: F-statistic and P-value.
    """
    try:
        formula = f'{dependent_var} ~ C({group_var})'
        model = ols(formula, data=data).fit()
        anova_results = anova_lm(model, typ=2, robust='hc3')
        # Extracting the F-statistic and p-value
        f_stat = anova_results.iloc[0]['F']
        p_value = anova_results.iloc[0]['PR(>F)']
        return f_stat, p_value
    except KeyError as e:
        print(f"Error: Missing column in data for ANOVA: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during Welch's ANOVA: {e}")
        raise

def plot_boxplot(data: DataFrame, x: str, y: str, hue: str, title: str, filename: str) -> None:
    """Create and save a boxplot."""
    try:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data, x=x, y=y, hue=hue)
        plt.title(title)
        plt.ylabel(y)
        plt.xlabel(x)
        plt.legend(title=hue)
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
    except Exception as e:
        print(f"Error while creating the boxplot: {e}")
        raise

def plot_interaction(data: DataFrame, x: str, y: str, hue: str, title: str, filename: str) -> None:
    """Create and save an interaction plot."""
    try:
        plt.figure(figsize=(10, 6))
        sns.pointplot(data=data, x=x, y=y, hue=hue, markers=['o', 's'], errorbar='sd')
        plt.title(title)
        plt.ylabel(y)
        plt.xlabel(x)
        plt.legend(title=hue)
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
    except Exception as e:
        print(f"Error while creating the interaction plot: {e}")
        raise

def main() -> None:
    """Main function to perform data analysis."""
    try:
        # Load data
        data_path = './src/cleaned_relevant_data.csv'
        data = load_data(data_path)

        print("Dataset loaded successfully. Here's a preview:")
        print(data.head())

        # --- Question 1: BDNF_N Analysis ---
        bdnf_data = data[['Genotype', 'Treatment', 'BDNF_N']].dropna()
        print("\n--- Question 1: BDNF_N Analysis ---")

        # Check normality
        check_normality(bdnf_data, 'BDNF_N', 'Treatment')

        # Check homogeneity of variances
        check_homogeneity(bdnf_data, 'BDNF_N', 'Treatment')

        # Perform Welch's ANOVA
        print("\n--- Performing Welch's ANOVA ---")
        bdnf_anova_results = perform_welchs_anova(bdnf_data, 'BDNF_N', 'Treatment')
        print(bdnf_anova_results)

        plot_boxplot(
            bdnf_data,
            x='Treatment',
            y='BDNF_N',
            hue='Genotype',
            title='BDNF_N Levels by Treatment and Genotype',
            filename='BDNF_N_boxplot.png'
        )

        # --- Question 2: pCREB_N Analysis ---
        pcreb_data = data[['Genotype', 'Treatment', 'pCREB_N']].dropna()
        pcreb_formula = 'pCREB_N ~ C(Genotype) * C(Treatment)'
        pcreb_model = ols(pcreb_formula, data=pcreb_data).fit()
        pcreb_anova_table = anova_lm(pcreb_model, typ=2)
        print("\n--- Question 2: pCREB_N Analysis ---")
        print(pcreb_anova_table)

        plot_interaction(
            pcreb_data,
            x='Treatment',
            y='pCREB_N',
            hue='Genotype',
            title='Interaction Effect: pCREB_N Levels by Treatment and Genotype',
            filename='pCREB_N_interaction_plot.png'
        )

    except Exception as e:
        print(f"An error occurred in the main function: {e}")

if __name__ == "__main__":
    main()
