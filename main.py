# Import the dataset and related variables directly from the analysis script
import sys
import os
sys.path.append(os.path.abspath('./src'))
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
    perform_two_way_anova,
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

    # Filter the data to include only relevant columns and drop rows with missing values
    pcreb_data = cleaned_data[['Genotype', 'Treatment', 'pCREB_N']].dropna()

    # Perform Two-Way ANOVA to analyze the interaction between Genotype and Treatment on pCREB_N levels
    pcreb_anova_table = perform_two_way_anova(pcreb_data, 'pCREB_N', 'Genotype', 'Treatment')

    # Print the ANOVA results table
    print("Two-Way ANOVA Results for pCREB_N:\n", pcreb_anova_table)

    # Create and save the interaction plot
    plot_interaction(
    pcreb_data,  # Data to visualize
    x='Treatment',  # Variable for the x-axis
    y='pCREB_N',  # Dependent variable for the y-axis
    hue='Genotype',  # Variable for grouping (hue)
    title='Interaction Effect: pCREB_N Levels by Treatment and Genotype',  # Plot title
    filename='pCREB_N_interaction_plot.png'  # File name for saving the plot
)

    # Final Conclusion
    print("\n=== Final Conclusion ===")
    if bdnf_p_value < 0.05:
        print("The treatment significantly affects BDNF_N levels (p < 0.05).")
    else:
        print("The treatment does not significantly affect BDNF_N levels (p >= 0.05).")

    if pcreb_anova_table["PR(>F)"]["C(Genotype):C(Treatment)"] < 0.05:
        print("There is a significant interaction effect between genotype and treatment on pCREB_N levels.")
    else:
        print("There is no significant interaction effect between genotype and treatment on pCREB_N levels.")

    print("\n=== Process Completed Successfully ===")


if __name__ == "__main__":
    main()


