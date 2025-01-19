
import pandas as pd
from pandas import DataFrame
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
import statsmodels.api as sm
import seaborn as sns
from typing import Tuple

# Set style for visualizations
sns.set(style="whitegrid", palette="muted")

def load_data(file_path: str) -> DataFrame:
    """Load the dataset from the specified file path."""
    return pd.read_csv(file_path)


def perform_anova(data: pd.DataFrame, column: str, group_col: str) -> Tuple[float, float]:
    """
    Perform ANOVA on a specified column.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data.
        column (str): Column to perform ANOVA on.
        group_col (str): Column specifying the groups.
    
    Returns:
        Tuple[float, float]: F-statistic and p-value from ANOVA.
    """
    groups = [group[column].dropna() for name, group in data.groupby(group_col)]
    f_stat, p_value = f_oneway(*groups)
    return f_stat, p_value


def plot_boxplot(data: DataFrame, x: str, y: str, hue: str, title: str, filename: str) -> None:
    """Create and save a boxplot."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x=x, y=y, hue=hue)
    plt.title(title)
    plt.ylabel(y)
    plt.xlabel(x)
    plt.legend(title=hue)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_interaction(data: DataFrame, x: str, y: str, hue: str, title: str, filename: str) -> None:
    """Create and save an interaction plot."""
    plt.figure(figsize=(10, 6))
    sns.pointplot(data=data, x=x, y=y, hue=hue, markers=['o', 's'], errorbar='sd')
    plt.title(title)
    plt.ylabel(y)
    plt.xlabel(x)
    plt.legend(title=hue)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def main() -> None:
    """Main function to perform data analysis."""
    # Load data
    data_path = './src/cleaned_relevant_data.csv'
    data = load_data(data_path)

    print("Dataset loaded successfully. Here's a preview:")
    print(data.head())

    # --- Question 1: BDNF_N Analysis ---
    bdnf_data = data[['Genotype', 'Treatment', 'BDNF_N']].dropna()
    print("\n--- Question 1: BDNF_N Analysis ---")
    bdnf_f_stat, bdnf_p_value = perform_anova(bdnf_data, 'BDNF_N', 'Treatment')
    print(f"ANOVA Results: F-statistic = {bdnf_f_stat:.4f}, P-value = {bdnf_p_value:.4f}")


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
    pcreb_anova_table = sm.stats.anova_lm(pcreb_model, typ=2)
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

if __name__ == "__main__":
    main()
