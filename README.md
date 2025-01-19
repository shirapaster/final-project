# Final Project: Data Analysis for Protein Levels

## Overview
This project investigates the impact of treatment (Saline or Memantine) on the expression levels of two proteins, BDNF_N and pCREB_N, in healthy and sick mice. This analysis aims to understand whether treatments have significant effects on these proteins and whether the effects vary by genotype.

### Key Questions
1. **Effect of Treatment on BDNF_N Levels:**
   - Does the treatment significantly affect BDNF_N levels in healthy and sick mice?

2. **Interaction Between Treatment and Genotype for pCREB_N Levels:**
   - Does the effect of treatment differ between healthy and sick mice for pCREB_N levels?
#להוסיף את שאלת המחקר הגדולה
---

## Project Structure
The project is structured as follows:

```
final-project/
├── src/
│   ├── data_analysis.py       # Main script for statistical analysis
│   ├── data_cleaning.py       # Script for cleaning and preprocessing the dataset
│   ├── cleaned_relevant_data.csv # Cleaned dataset used for analysis
│   └── Data_Cortex_Nuclear.csv  # Original dataset provided for cleaning
├── test/
│   ├── test_analysis.py       # Unit tests for the statistical analysis
│   └── test_cleaning.py       # Unit tests for data cleaning and preprocessing
├── README.md                  # Project documentation (this file)
└── requirements.txt           # Python dependencies
```

---

## Setup and Installation

### Prerequisites
- Python 3.10 or higher
- Ensure `pip` is installed on your system.

### Installation Steps
1. Clone the repository:
   ```power shell
   git clone <repository_url>
   cd final-project
   ```

2. Install the required dependencies:
   ```power shell
   pip install -r requirements.txt
   ```

3. Ensure the dataset `Data_Cortex_Nuclear.csv` is placed in the `src/` directory for cleaning, if needed.

---

## Running the Project

### Step 1: Data Cleaning
To preprocess the dataset, run the cleaning script:
```bash
python src/data_cleaning.py
```
This generates a cleaned dataset `cleaned_relevant_data.csv` in the `src/` directory.

### Step 2: Data Analysis
To perform statistical analysis, run the analysis script:
```bash
python src/data_analysis.py
```
The output includes:
- Statistical results (ANOVA) for BDNF_N and pCREB_N.
- Visualizations saved as:
  - `BDNF_N_boxplot.png`
  - `pCREB_N_interaction_plot.png`

---

## Unit Tests

### Overview
The project includes unit tests to validate the correctness of the cleaning and analysis scripts. These tests are located in the `test/` directory:
- `test_cleaning.py`: Validates the data cleaning process.
- `test_analysis.py`: Tests statistical calculations and visualizations.

### Running the Tests
To execute all tests, use the following command:
```bash
python -m unittest discover -s test
```

---

## Results and Conclusions

### Summary of Findings
1. **BDNF_N Levels:**
   - The treatment significantly affects BDNF_N levels (P < 0.05).

2. **pCREB_N Levels:**
   - The treatment and genotype have independent effects on pCREB_N levels.
   - No significant interaction was found between treatment and genotype for pCREB_N levels.

### General Conclusion
These findings indicate that treatment influences protein expression, but the effects vary depending on the protein and genotype. The results emphasize the need for personalized approaches in neuroscience research.

---

## Required Dependencies

To ensure the project runs smoothly, the following Python libraries must be installed:
- `pandas`: Data manipulation and cleaning.
- `scipy`: Statistical analysis.
- `statsmodels`: Advanced statistical modeling.
- `matplotlib`: Data visualization.
- `seaborn`: Enhanced visualizations.
- `unittest`: Testing framework.

To install all dependencies, run:
```power shell
pip install -r requirements.txt
```

---

## Future Work
1. Perform post-hoc analyses to explore specific group differences.
2. Extend the analysis to include additional datasets or proteins.
3. Improve visualizations by adding interactive plots and annotations.

---

## Authors
This project was collaboratively developed by two students as part of an academic requirement:

- **Shira**: Responsible for data analysis and visualizations.
- **eliana**: Focused on data cleaning and testing.

---

## Contact Information
For questions, feedback, or collaboration opportunities, please contact:
- **Email:** shirapaster@gmail.com
- **GitHub Repository:** https://github.com/shirapaster/final-project

