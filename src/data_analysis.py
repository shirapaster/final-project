# Importing necessary libraries
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for visualizations
sns.set(style="whitegrid", palette="muted")

# Step 1: Load the dataset
data_path = './src/cleaned_relevant_data.csv'
data = pd.read_csv(data_path)

# Display the first few rows to ensure data is loaded correctly
print("Dataset loaded successfully. Here's a preview:")
print(data.head())

# --- Question 1: Effect of treatment on BDNF_N levels in healthy vs. sick mice ---
# Research Context:
# - BDNF_N is critical for neuroplasticity, learning, and memory.
# - The question explores whether treatment significantly affects BDNF_N levels in healthy and sick mice.

# Filter data for BDNF_N analysis
bdnf_data = data[['Genotype', 'Treatment', 'BDNF_N']].dropna()

# Perform ANOVA to analyze treatment effect on BDNF_N
bdnf_anova = stats.f_oneway(
    bdnf_data[bdnf_data['Treatment'] == 'Saline']['BDNF_N'],
    bdnf_data[bdnf_data['Treatment'] == 'Memantine']['BDNF_N']
)

# Display the ANOVA results
print("\n--- Question 1: BDNF_N Analysis ---")
print(f"ANOVA Results: F-statistic = {bdnf_anova.statistic:.4f}, P-value = {bdnf_anova.pvalue:.4f}")

# Conclusion for Question 1
if bdnf_anova.pvalue < 0.05:
    print("Conclusion for Question 1: The treatment significantly affects BDNF_N levels.")
else:
    print("Conclusion for Question 1: The treatment does not significantly affect BDNF_N levels.")

# Visualization: Boxplot for BDNF_N
plt.figure(figsize=(10, 6))
sns.boxplot(data=bdnf_data, x='Treatment', y='BDNF_N', hue='Genotype')
plt.title('BDNF_N Levels by Treatment and Genotype')
plt.ylabel('BDNF_N Levels')
plt.xlabel('Treatment')
plt.legend(title='Genotype')
plt.tight_layout()
plt.savefig('BDNF_N_boxplot.png')
plt.show()

# --- Question 2: Does treatment effect on pCREB_N differ between healthy and sick mice? ---
# Research Context:
# - pCREB_N is a key regulator of gene expression and neuronal signaling.
# - The question examines the interaction between treatment and genotype.

# Filter data for pCREB_N analysis
pcreb_data = data[['Genotype', 'Treatment', 'pCREB_N']].dropna()

# Two-way ANOVA to analyze interaction effects
pcreb_formula = 'pCREB_N ~ C(Genotype) * C(Treatment)'
pcreb_model = ols(pcreb_formula, data=pcreb_data).fit()
pcreb_anova_table = sm.stats.anova_lm(pcreb_model, typ=2)

# Display the ANOVA table
print("\n--- Question 2: pCREB_N Analysis ---")
print(pcreb_anova_table)

# Conclusion for Question 2
interaction_p_value = pcreb_anova_table.loc['C(Genotype):C(Treatment)', 'PR(>F)']
if interaction_p_value < 0.05:
    print("Conclusion for Question 2: There is a significant interaction between treatment and genotype for pCREB_N levels.")
else:
    print("Conclusion for Question 2: There is no significant interaction between treatment and genotype for pCREB_N levels.")

# Visualization: Interaction plot for pCREB_N
plt.figure(figsize=(10, 6))
sns.pointplot(data=pcreb_data, x='Treatment', y='pCREB_N', hue='Genotype', markers=['o', 's'], errorbar='sd')
plt.title('Interaction Effect: pCREB_N Levels by Treatment and Genotype')
plt.ylabel('pCREB_N Levels')
plt.xlabel('Treatment')
plt.legend(title='Genotype')
plt.tight_layout()
plt.savefig('pCREB_N_interaction_plot.png')
plt.show()

# --- General Research Question: How does treatment influence protein levels in mice? ---
# General Conclusion:
print("\n--- General Conclusion ---")
bdnf_conclusion = (
    "The treatment significantly affects BDNF_N levels."
    if bdnf_anova.pvalue < 0.05 else
    "The treatment does not significantly affect BDNF_N levels."
)
pcreb_conclusion = (
    "There is a significant interaction between treatment and genotype for pCREB_N levels."
    if interaction_p_value < 0.05 else
    "There is no significant interaction between treatment and genotype for pCREB_N levels."
)

print(
    "This analysis investigated the impact of treatment (Saline or Memantine) on two key proteins in healthy and sick mice:\n"
    f"1. BDNF_N: {bdnf_conclusion}\n"
    f"2. pCREB_N: {pcreb_conclusion}\n"
    "These findings suggest that treatment may influence protein expression differently depending on genotype, "
    "emphasizing the importance of personalized treatment approaches in neuroscience."
)
