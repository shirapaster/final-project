# Import the dataset and related variables directly from the analysis script
import sys
import os
sys.path.append(os.path.abspath('./src'))
from statsmodels.formula.api import ols
import statsmodels.api as sm
from data_cleaning import (
    load_data,
    drop_columns_with_many_missing,
    fill_missing_values_by_group,
    remove_outliers,
    save_cleaned_data,
    check_group_balance
)
from data_analysis import (
    perform_welchs_anova,
    plot_boxplot,
    plot_interaction
)

def main() -> None:
    """
    Main function to orchestrate the data cleaning and analysis process.
    This script loads raw data, cleans it, performs analysis, and summarizes results.
    """

    # Step 1: Define paths
    raw_data_path: str = "./src/Data_Cortex_Nuclear.csv"  # Path to the raw data file
    cleaned_data_path: str = "./src/cleaned_relevant_data.csv"  # Path to the cleaned data file

    # Step 2: Load the raw data
    print("=== Loading Data ===")
    data = load_data(raw_data_path)

    # Step 3: Inspect and clean the data
    print("\n=== Inspecting and Cleaning Data ===")
    data = drop_columns_with_many_missing(data, threshold=0.5)
    data = fill_missing_values_by_group(data, ['BDNF_N', 'pCREB_N'], ['Genotype', 'Treatment'])
    data = remove_outliers(data, ['BDNF_N', 'pCREB_N'], factor=3)

    # Step 4: Check group balance
    print("\n=== Checking Group Balance ===")
    check_group_balance(data)

    # Step 5: Save the cleaned data
    print("\n=== Saving Cleaned Data ===")
    save_cleaned_data(data, cleaned_data_path)

    # Step 6: Perform data analysis
    print("\n=== Starting Data Analysis ===")

    # Load cleaned data for analysis
    cleaned_data = load_data(cleaned_data_path)

    # Question 1: Analyze BDNF_N
    print("\n--- Question 1: BDNF_N Analysis ---")
    bdnf_data = cleaned_data[['Genotype', 'Treatment', 'BDNF_N']].dropna()
    bdnf_f_stat, bdnf_p_value = perform_welchs_anova(bdnf_data, 'BDNF_N', 'Treatment')


    print(f"ANOVA Results: F-statistic = {bdnf_f_stat:.4f}, P-value = {bdnf_p_value:.4f}")

    plot_boxplot(
        bdnf_data,
        x='Treatment',
        y='BDNF_N',
        hue='Genotype',
        title='BDNF_N Levels by Treatment and Genotype',
        filename='BDNF_N_boxplot.png'
    )

    # Question 2: Analyze pCREB_N
    print("\n--- Question 2: pCREB_N Analysis ---")
    pcreb_data = cleaned_data[['Genotype', 'Treatment', 'pCREB_N']].dropna()
    pcreb_formula = 'pCREB_N ~ C(Genotype) * C(Treatment)'
    pcreb_model = ols(pcreb_formula, data=pcreb_data).fit()
    pcreb_anova_table = sm.stats.anova_lm(pcreb_model, typ=2)
    print(pcreb_anova_table)

    plot_interaction(
        pcreb_data,
        x='Treatment',
        y='pCREB_N',
        hue='Genotype',
        title='Interaction Effect: pCREB_N Levels by Treatment and Genotype',
        filename='pCREB_N_interaction_plot.png'
    )

    print("\n=== Process Completed Successfully ===")

if __name__ == "__main__":
    main()


