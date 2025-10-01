# Cu-Fe Database Data Processing Summary

## Overview
This directory contains the data processing pipeline for the Cu-Fe alloy database, focusing on compressive strength prediction. The original dataset has been cleaned, analyzed, and prepared for machine learning modeling.

## Files Description

### 1. `clean_cu_fe_data.py`
Main data cleaning script that:
- Removes features with >70% missing values (Yield Strength, Ultimate Tensile Strength, Elongation)
- Extracts temperature and processing method information from text notes
- Creates derived features (composition ratios, element counts, etc.)
- Handles missing values and infinite values
- Encodes categorical variables

### 2. `data_analysis.py`
Comprehensive analysis script that:
- Performs exploratory data analysis
- Calculates feature correlations and importance
- Provides modeling recommendations
- Generates feature selection insights

### 3. `Cu_Fe_DB_cleaned.csv`
The cleaned dataset ready for modeling with:
- **105 samples** (no missing target values)
- **60 features** (including derived features + Source column)
- **No missing or infinite values**
- **DOI links preserved** in Source column for research traceability

### 4. `feature_analysis_summary.csv`
Detailed feature analysis with correlation, importance, and composite scores

### 5. `data_utils.py`
Utility functions for working with the cleaned dataset including DOI reference handling

### 6. `source_reference_guide.csv`
Complete mapping of compositions to their DOI sources for research traceability

## Key Findings

### Dataset Characteristics
- **Target Variable**: Compressive Strength (MPa)
  - Range: 140 - 3,620 MPa
  - Mean: 1,287 ± 694 MPa
  - Distribution: Slightly right-skewed (0.846)

### Most Important Features for Compressive Strength Prediction

#### Top 15 Features (Composite Ranking):
1. **Hardness (HVN)** - Strongest predictor (correlation: 0.58)
2. **BCC Crystal Structure** - Strong correlation (0.50)
3. **Has_BCC_Phase** - Phase indicator (0.50)
4. **Method_Unknown** - Processing method (0.44)
5. **Method_arc_melting** - Arc melting process (0.42)
6. **Mo (Molybdenum)** - Element content (0.35)
7. **Plasticity (%) - Compressive** - Mechanical property (0.33)
8. **Method_sintering** - Sintering process (0.32)
9. **Num_Elements** - Number of alloying elements (0.30)
10. **Num_Phases** - Phase complexity (0.29)
11. **Num_Crystal_Structures** - Structural complexity (0.28)
12. **Cu_Fe_Ratio** - Composition ratio (0.27)
13. **V (Vanadium)** - Element content (0.26)
14. **Zn (Zinc)** - Element content (0.26)
15. **Has_FCC_Phase** - Phase indicator (0.25)

### Composition Analysis
- **Cu**: Present in all 105 samples (avg: 16.6%)
- **Ni**: Present in 89 samples (avg: 20.6%)
- **Fe**: Present in 84 samples (avg: 17.8%)
- **Al**: Present in 68 samples (avg: 18.0%)
- **Co**: Present in 68 samples (avg: 17.9%)

### Processing Methods Distribution
- Unknown: 52 samples (49.5%)
- Arc melting: 15 samples (14.3%)
- Annealing: 12 samples (11.4%)
- Sintering: 11 samples (10.5%)
- Other methods: 15 samples (14.3%)

### Crystal Structure Distribution
- **FCC**: 86 samples (81.9%)
- **BCC**: 39 samples (37.1%)
- **HCP**: 3 samples (2.9%)
- **IM (Intermetallic)**: 20 samples (19.0%)

### Source Distribution (DOI Analysis)
- **16 unique research sources** with DOI links preserved
- **Top 5 sources by sample count:**
  - 28 samples: https://doi.org/10.1016/j.dib.2018.11.111
  - 15 samples: https://doi.org/10.1016/j.vacuum.2020.109173  
  - 11 samples: dx.doi.org/10.1016/j.apsusc.2015.07.207
  - 9 samples: https://doi.org/10.1016/j.jallcom.2022.164310
  - 9 samples: http://dx.doi.org/10.1016/j.jallcom.2014.06.090

## Data Quality Assessment

### Strengths
✅ **Complete target data**: No missing compressive strength values  
✅ **Balanced compositions**: All composition totals ≈ 1.0  
✅ **No infinite values**: Successfully handled division by zero  
✅ **Rich feature set**: 59 features including derived variables  
✅ **Multiple data types**: Mechanical, compositional, and processing features  

### Challenges
⚠️ **Small dataset**: Only 105 samples for 59 features  
⚠️ **High dimensionality**: Feature-to-sample ratio of 0.55  
⚠️ **Missing processing info**: 49% of samples have unknown processing  
⚠️ **Duplicate data**: 1 duplicate row identified  
⚠️ **Potential outliers**: 3 samples identified as statistical outliers  

## Modeling Recommendations

### Recommended Algorithms
1. **Random Forest Regressor** - Handles non-linearity and feature interactions well
2. **Gradient Boosting** (XGBoost/LightGBM) - Excellent for small datasets
3. **Support Vector Regression** with RBF kernel - Good for high-dimensional data
4. **Ridge/Lasso Regression** - Provides regularization for small datasets
5. **Ensemble Methods** - Combine multiple algorithms for robustness

### Cross-Validation Strategy
- **5-fold or 10-fold CV** due to small sample size
- **Stratified sampling** based on processing method if possible
- **Nested CV** for hyperparameter tuning to avoid overfitting

### Feature Selection
- Use top 15-20 features based on composite ranking
- Consider regularization techniques (L1/L2)
- Apply recursive feature elimination with cross-validation

### Data Preprocessing
- **Standardization**: Recommended for SVM and neural networks
- **Log transformation**: Consider for target variable (slightly skewed)
- **Outlier handling**: Investigate the 3 identified outliers

## Expected Model Performance
Based on Random Forest baseline:
- **R² Score**: ~0.79 (good predictive power)
- **RMSE**: ~380 MPa (reasonable for the data range)

## Usage Instructions

### 1. Install Dependencies
```bash
pip install -r data_processing/requirements.txt
```

### 2. Run Data Cleaning
```bash
python data_processing/clean_cu_fe_data.py
```

### 3. Run Data Analysis
```bash
python data_processing/data_analysis.py
```

### 4. Use Data Utilities (with DOI support)
```bash
python data_processing/data_utils.py
```

### 5. Load Data for Modeling
```python
from data_processing.data_utils import load_data_for_modeling, get_top_features

# Load cleaned data
X, y, sources = load_data_for_modeling()

# Get top features for modeling
top_features = get_top_features(15)
X_selected = X[top_features]

# Access DOI information
print("Sample sources:", sources.head())
```

## Next Steps for Modeling
1. **Feature Selection**: Use top 15-20 features identified
2. **Model Training**: Implement recommended algorithms with cross-validation
3. **Hyperparameter Tuning**: Use grid search or Bayesian optimization
4. **Ensemble Methods**: Combine best performing models
5. **Model Interpretation**: Use SHAP or LIME for feature importance analysis

## Contact & Notes
- The cleaned dataset is ready for immediate use in machine learning pipelines
- All preprocessing steps are documented and reproducible
- Feature engineering has been optimized for compressive strength prediction
- Consider collecting more data to improve model robustness