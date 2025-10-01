"""
Analysis Script for Merged Database
Analyzes the combined Cu-Fe database with the new database
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

def load_merged_database(file_path: str = "data_processing/merged_database_cleaned.csv") -> pd.DataFrame:
    """Load the merged database."""
    try:
        df = pd.read_csv(file_path)
        print(f"Merged database loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading merged database: {e}")
        return None

def compare_databases() -> None:
    """Compare the original, new, and merged databases."""
    print("\n=== Database Comparison ===")
    
    try:
        # Load all databases
        original_df = pd.read_csv("data_processing/Cu_Fe_DB_cleaned.csv")
        new_df = pd.read_csv("data_processing/new_database_cleaned.csv")
        merged_df = pd.read_csv("data_processing/merged_database_cleaned.csv")
        
        print(f"Original Cu-Fe Database: {original_df.shape}")
        print(f"New Database (cleaned): {new_df.shape}")
        print(f"Merged Database: {merged_df.shape}")
        
        # Compare target variable statistics
        print(f"\nCompressive Strength Statistics:")
        
        orig_target = original_df['Compressive strength (MPa)']
        new_target = new_df['Compressive strength (MPa)']
        merged_target = merged_df['Compressive strength (MPa)']
        
        stats_df = pd.DataFrame({
            'Original': [orig_target.count(), orig_target.mean(), orig_target.std(), orig_target.min(), orig_target.max()],
            'New': [new_target.count(), new_target.mean(), new_target.std(), new_target.min(), new_target.max()],
            'Merged': [merged_target.count(), merged_target.mean(), merged_target.std(), merged_target.min(), merged_target.max()]
        }, index=['Count', 'Mean', 'Std', 'Min', 'Max'])
        
        print(stats_df.round(2))
        
    except Exception as e:
        print(f"Error in database comparison: {e}")

def analyze_element_distribution(df: pd.DataFrame) -> None:
    """Analyze element distribution in the merged database."""
    print("\n=== Element Distribution Analysis ===")
    
    # Get element columns
    element_cols = ['Al', 'C', 'Co', 'Cr', 'Cu', 'Fe', 'Ga', 'Gd', 'Mg', 'Mn', 'Mo', 'Ni', 'Pb', 'Ti', 'V', 'Zn', 'Zr']
    existing_elements = [col for col in element_cols if col in df.columns]
    
    print(f"Elements in merged database: {existing_elements}")
    
    # Calculate element statistics
    element_stats = []
    for element in existing_elements:
        non_zero_count = (df[element] > 0).sum()
        if non_zero_count > 0:
            mean_content = df[df[element] > 0][element].mean()
            max_content = df[element].max()
            element_stats.append({
                'Element': element,
                'Samples_Present': non_zero_count,
                'Percentage_Present': (non_zero_count / len(df)) * 100,
                'Mean_Content': mean_content,
                'Max_Content': max_content
            })
    
    element_stats_df = pd.DataFrame(element_stats)
    element_stats_df = element_stats_df.sort_values('Samples_Present', ascending=False)
    
    print(f"\nElement Presence and Content:")
    print(element_stats_df.round(3))

def analyze_source_distribution(df: pd.DataFrame) -> None:
    """Analyze source distribution in the merged database."""
    print("\n=== Source Distribution Analysis ===")
    
    if 'Source' in df.columns:
        source_stats = df.groupby('Source').agg({
            'Compressive strength (MPa)': ['count', 'mean', 'std', 'min', 'max']
        }).round(2)
        
        source_stats.columns = ['Count', 'Mean_Strength', 'Std_Strength', 'Min_Strength', 'Max_Strength']
        source_stats = source_stats.sort_values('Count', ascending=False)
        
        print(f"Top 10 sources by sample count:")
        print(source_stats.head(10))
        
        print(f"\nTotal unique sources: {df['Source'].nunique()}")

def feature_importance_merged(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze feature importance for the merged database."""
    print("\n=== Feature Importance Analysis (Merged Database) ===")
    
    # Prepare data for modeling
    exclude_cols = ['Compressive strength (MPa)', 'Source', 'Unique ID', 'Composition', 'Phases present']
    
    # Get only numeric columns for modeling
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['Compressive strength (MPa)']
    
    # Handle any remaining missing values
    X = X.fillna(X.median())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("Top 20 Most Important Features (Merged Database):")
    print(importance_df.head(20))
    
    # Model performance
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nRandom Forest Model Performance (Merged Database):")
    print(f"R² Score: {r2:.3f}")
    print(f"RMSE: {np.sqrt(mse):.2f} MPa")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    return importance_df

def correlation_analysis_merged(df: pd.DataFrame) -> None:
    """Analyze correlations in the merged database."""
    print("\n=== Correlation Analysis (Merged Database) ===")
    
    # Calculate correlations with target
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    target_corr = df[numeric_cols].corr()['Compressive strength (MPa)'].abs().sort_values(ascending=False)
    
    print("Top 15 features correlated with Compressive Strength:")
    print(target_corr.head(16)[1:])  # Exclude self-correlation

def data_quality_merged(df: pd.DataFrame) -> None:
    """Assess data quality of the merged database."""
    print("\n=== Data Quality Assessment (Merged Database) ===")
    
    # Basic statistics
    print(f"Total samples: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    
    # Missing values
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    print(f"Total missing values: {total_missing}")
    
    if total_missing > 0:
        print("Columns with missing values:")
        for col, count in missing_counts.items():
            if count > 0:
                print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    
    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")
    
    # Target variable quality
    target = df['Compressive strength (MPa)']
    print(f"\nTarget Variable Quality:")
    print(f"Range: {target.min():.1f} - {target.max():.1f} MPa")
    print(f"Mean ± Std: {target.mean():.1f} ± {target.std():.1f} MPa")
    print(f"Coefficient of Variation: {target.std()/target.mean():.3f}")
    
    # Check for outliers
    Q1 = target.quantile(0.25)
    Q3 = target.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((target < lower_bound) | (target > upper_bound)).sum()
    print(f"Potential outliers: {outliers} ({outliers/len(df)*100:.1f}%)")

def generate_merged_recommendations(df: pd.DataFrame, importance_df: pd.DataFrame) -> None:
    """Generate recommendations for modeling with the merged database."""
    print("\n=== Modeling Recommendations (Merged Database) ===")
    
    # Dataset characteristics
    n_samples = len(df)
    n_features = len(df.columns) - 1  # Exclude target
    
    print(f"Dataset Characteristics:")
    print(f"- Sample size: {n_samples}")
    print(f"- Feature count: {n_features}")
    print(f"- Feature-to-sample ratio: {n_features/n_samples:.3f}")
    
    # Recommended features
    top_features = importance_df.head(25)['Feature'].tolist()
    print(f"\nTop 25 Recommended Features:")
    for i, feature in enumerate(top_features, 1):
        print(f"{i:2d}. {feature}")
    
    # Modeling strategy
    print(f"\nRecommended Modeling Strategy:")
    if n_samples > 200:
        print("✓ Good sample size for complex models")
        print("- Try ensemble methods (Random Forest, XGBoost, LightGBM)")
        print("- Neural networks with regularization")
        print("- Support Vector Regression")
    else:
        print("- Moderate sample size - use regularization")
        print("- Random Forest with limited depth")
        print("- Ridge/Lasso regression")
        print("- Gradient boosting with early stopping")
    
    print(f"\nCross-Validation Strategy:")
    if n_samples > 200:
        print("- Use 10-fold cross-validation")
        print("- Consider stratified sampling by source")
    else:
        print("- Use 5-fold cross-validation")
        print("- Nested CV for hyperparameter tuning")
    
    # Feature engineering suggestions
    print(f"\nFeature Engineering Suggestions:")
    print("- Consider polynomial features for top elements")
    print("- Element interaction terms (e.g., Cu*Fe, Al*Ni)")
    print("- Ratio features (already included: Cu/Fe ratio)")
    print("- Log transformation for highly skewed features")

def save_merged_analysis(importance_df: pd.DataFrame) -> None:
    """Save analysis results."""
    try:
        importance_df.to_csv("data_processing/merged_feature_importance.csv", index=False)
        print(f"\nMerged database feature importance saved to: data_processing/merged_feature_importance.csv")
    except Exception as e:
        print(f"Error saving analysis: {e}")

def main():
    """Main analysis pipeline for merged database."""
    print("=== Merged Database Analysis Pipeline ===")
    
    # Load merged database
    df = load_merged_database()
    if df is None:
        return
    
    # Compare databases
    compare_databases()
    
    # Analyze element distribution
    analyze_element_distribution(df)
    
    # Analyze source distribution
    analyze_source_distribution(df)
    
    # Feature importance analysis
    importance_df = feature_importance_merged(df)
    
    # Correlation analysis
    correlation_analysis_merged(df)
    
    # Data quality assessment
    data_quality_merged(df)
    
    # Generate recommendations
    generate_merged_recommendations(df, importance_df)
    
    # Save results
    save_merged_analysis(importance_df)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Merged database ready for advanced modeling!")

if __name__ == "__main__":
    main()