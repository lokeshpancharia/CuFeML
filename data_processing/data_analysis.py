"""
Data Analysis Script for Cleaned Cu-Fe Database
Provides detailed analysis of the cleaned dataset for compressive strength modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_cleaned_data(file_path: str) -> pd.DataFrame:
    """Load the cleaned dataset."""
    try:
        df = pd.read_csv(file_path)
        print(f"Cleaned data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading cleaned data: {e}")
        return None

def exploratory_data_analysis(df: pd.DataFrame) -> None:
    """Perform exploratory data analysis."""
    print("\n=== Exploratory Data Analysis ===")
    
    # Basic statistics
    print("\nDataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # Target variable analysis
    target = df['Compressive strength (MPa)']
    print(f"\nTarget Variable Statistics:")
    print(f"Mean: {target.mean():.2f} MPa")
    print(f"Median: {target.median():.2f} MPa")
    print(f"Std: {target.std():.2f} MPa")
    print(f"Min: {target.min():.2f} MPa")
    print(f"Max: {target.max():.2f} MPa")
    print(f"Skewness: {target.skew():.3f}")
    print(f"Kurtosis: {target.kurtosis():.3f}")
    
    # Composition analysis
    element_cols = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Mo', 'Ni', 'Ti', 'Si', 'V', 'Zn', 'Zr']
    existing_elements = [col for col in element_cols if col in df.columns]
    
    print(f"\nComposition Analysis:")
    print(f"Elements present: {existing_elements}")
    
    for element in existing_elements:
        non_zero_count = (df[element] > 0).sum()
        if non_zero_count > 0:
            mean_content = df[df[element] > 0][element].mean()
            print(f"{element}: Present in {non_zero_count} samples, avg content: {mean_content:.3f}")
    
    # Processing method distribution
    method_cols = [col for col in df.columns if col.startswith('Method_')]
    if method_cols:
        print(f"\nProcessing Methods Distribution:")
        for method in method_cols:
            count = df[method].sum()
            if count > 0:
                print(f"{method.replace('Method_', '')}: {count} samples")
    
    # Crystal structure analysis
    structure_cols = ['FCC', 'BCC', 'HCP', 'IM']
    existing_structures = [col for col in structure_cols if col in df.columns]
    
    if existing_structures:
        print(f"\nCrystal Structure Distribution:")
        for structure in existing_structures:
            count = df[structure].sum()
            print(f"{structure}: {count} samples")

def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze correlations with target variable."""
    print("\n=== Correlation Analysis ===")
    
    # Calculate correlations with target
    target_corr = df.corr()['Compressive strength (MPa)'].abs().sort_values(ascending=False)
    
    print("Top 15 features correlated with Compressive Strength:")
    print(target_corr.head(16)[1:])  # Exclude self-correlation
    
    return target_corr

def feature_importance_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze feature importance using Random Forest."""
    print("\n=== Feature Importance Analysis ===")
    
    # Prepare data
    X = df.drop(['Compressive strength (MPa)'], axis=1)
    y = df['Compressive strength (MPa)']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("Top 15 Most Important Features (Random Forest):")
    print(importance_df.head(15))
    
    # Model performance
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nRandom Forest Model Performance:")
    print(f"R² Score: {r2:.3f}")
    print(f"RMSE: {np.sqrt(mse):.2f} MPa")
    
    return importance_df

def statistical_feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    """Perform statistical feature selection."""
    print("\n=== Statistical Feature Selection ===")
    
    # Prepare data
    X = df.drop(['Compressive strength (MPa)'], axis=1)
    y = df['Compressive strength (MPa)']
    
    # Select top k features using f_regression
    selector = SelectKBest(score_func=f_regression, k=15)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected features
    selected_features = X.columns[selector.get_support()]
    feature_scores = selector.scores_[selector.get_support()]
    
    statistical_df = pd.DataFrame({
        'Feature': selected_features,
        'F_Score': feature_scores
    }).sort_values('F_Score', ascending=False)
    
    print("Top 15 Features (Statistical Selection):")
    print(statistical_df)
    
    return statistical_df

def data_quality_report(df: pd.DataFrame) -> None:
    """Generate data quality report."""
    print("\n=== Data Quality Report ===")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")
    
    # Check data types
    print(f"\nData Types:")
    print(df.dtypes.value_counts())
    
    # Check for infinite values
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    print(f"Infinite values: {inf_count}")
    
    # Check composition totals
    element_cols = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Mo', 'Ni', 'Ti', 'Si', 'V', 'Zn', 'Zr']
    existing_elements = [col for col in element_cols if col in df.columns]
    
    if 'Total_Composition' in df.columns:
        comp_stats = df['Total_Composition'].describe()
        print(f"\nComposition Total Statistics:")
        print(comp_stats)
        
        # Check for compositions significantly different from 1.0
        unusual_comp = ((df['Total_Composition'] < 0.8) | (df['Total_Composition'] > 1.2)).sum()
        print(f"Samples with unusual composition totals (< 0.8 or > 1.2): {unusual_comp}")

def create_feature_summary(df: pd.DataFrame, corr_df: pd.DataFrame, 
                          importance_df: pd.DataFrame, statistical_df: pd.DataFrame) -> pd.DataFrame:
    """Create comprehensive feature summary."""
    print("\n=== Feature Summary ===")
    
    # Combine all analyses
    feature_summary = pd.DataFrame({'Feature': df.columns})
    
    # Add correlation info
    feature_summary = feature_summary.merge(
        corr_df.reset_index().rename(columns={'index': 'Feature', 'Compressive strength (MPa)': 'Correlation'}),
        on='Feature', how='left'
    )
    
    # Add importance info
    feature_summary = feature_summary.merge(
        importance_df[['Feature', 'Importance']], on='Feature', how='left'
    )
    
    # Add statistical info
    feature_summary = feature_summary.merge(
        statistical_df[['Feature', 'F_Score']], on='Feature', how='left'
    )
    
    # Fill NaN values
    feature_summary = feature_summary.fillna(0)
    
    # Create composite score
    feature_summary['Composite_Score'] = (
        feature_summary['Correlation'] * 0.3 +
        feature_summary['Importance'] * 0.4 +
        (feature_summary['F_Score'] / feature_summary['F_Score'].max()) * 0.3
    )
    
    # Sort by composite score
    feature_summary = feature_summary.sort_values('Composite_Score', ascending=False)
    
    print("Top 20 Features (Composite Ranking):")
    print(feature_summary.head(20)[['Feature', 'Correlation', 'Importance', 'Composite_Score']])
    
    return feature_summary

def generate_recommendations(df: pd.DataFrame, feature_summary: pd.DataFrame) -> None:
    """Generate modeling recommendations."""
    print("\n=== Modeling Recommendations ===")
    
    # Top features for modeling
    top_features = feature_summary.head(20)['Feature'].tolist()
    if 'Compressive strength (MPa)' in top_features:
        top_features.remove('Compressive strength (MPa)')
    
    print(f"Recommended features for modeling ({len(top_features)}):")
    for i, feature in enumerate(top_features[:15], 1):
        print(f"{i:2d}. {feature}")
    
    # Data characteristics
    print(f"\nDataset Characteristics:")
    print(f"- Sample size: {len(df)} (small dataset - consider regularization)")
    print(f"- Feature count: {len(df.columns)-1}")
    print(f"- Feature-to-sample ratio: {(len(df.columns)-1)/len(df):.2f}")
    
    if len(df.columns) > len(df) * 0.1:
        print("- High dimensionality relative to sample size - feature selection recommended")
    
    # Target variable characteristics
    target = df['Compressive strength (MPa)']
    if target.skew() > 1:
        print("- Target variable is right-skewed - consider log transformation")
    elif target.skew() < -1:
        print("- Target variable is left-skewed - consider transformation")
    
    print(f"\nRecommended modeling approaches:")
    print("1. Random Forest Regressor (handles non-linearity well)")
    print("2. Gradient Boosting (XGBoost/LightGBM)")
    print("3. Support Vector Regression with RBF kernel")
    print("4. Ridge/Lasso Regression (for regularization)")
    print("5. Ensemble methods combining multiple algorithms")
    
    print(f"\nCross-validation strategy:")
    print("- Use 5-fold or 10-fold CV due to small sample size")
    print("- Consider stratified sampling based on processing method")
    print("- Use nested CV for hyperparameter tuning")

def main():
    """Main analysis pipeline."""
    print("=== Cu-Fe Database Analysis Pipeline ===")
    
    # Load cleaned data
    df = load_cleaned_data("data_processing/Cu_Fe_DB_cleaned.csv")
    if df is None:
        return
    
    # Perform analyses
    exploratory_data_analysis(df)
    corr_df = correlation_analysis(df)
    importance_df = feature_importance_analysis(df)
    statistical_df = statistical_feature_selection(df)
    
    # Create comprehensive summary
    feature_summary = create_feature_summary(df, corr_df, importance_df, statistical_df)
    
    # Generate recommendations
    generate_recommendations(df, feature_summary)
    
    # Save feature summary
    feature_summary.to_csv("data_processing/feature_analysis_summary.csv", index=False)
    print(f"\nFeature analysis summary saved to: data_processing/feature_analysis_summary.csv")
    
    # Data quality report
    data_quality_report(df)

if __name__ == "__main__":
    main()