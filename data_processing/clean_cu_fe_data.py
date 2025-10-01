"""
Data Cleaning Script for Cu-Fe Database
Target: Compressive Strength Prediction

This script cleans the Cu_Fe_DB.csv file by:
1. Dropping features with null values
2. Extracting temperature and processing method information
3. Feature engineering and data preprocessing
4. Creating a cleaned dataset for compressive strength modeling
"""

import pandas as pd
import numpy as np
import re
from typing import Tuple, Dict, List

def load_data(file_path: str) -> pd.DataFrame:
    """Load the Cu-Fe database CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def analyze_missing_data(df: pd.DataFrame) -> None:
    """Analyze missing data patterns."""
    print("\n=== Missing Data Analysis ===")
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    missing_info = pd.DataFrame({
        'Missing_Count': missing_counts,
        'Missing_Percentage': missing_percentages
    }).sort_values('Missing_Percentage', ascending=False)
    
    print(missing_info[missing_info['Missing_Count'] > 0])
    
    # Check target variable
    target_missing = df['Compressive strength (MPa)'].isnull().sum()
    print(f"\nTarget variable (Compressive strength) missing values: {target_missing}")

def extract_temperature_info(processing_note: str) -> Dict:
    """Extract temperature information from processing notes."""
    if pd.isna(processing_note):
        return {'temperature_celsius': None, 'temperature_kelvin': None}
    
    # Convert to string and make lowercase for easier matching
    note = str(processing_note).lower()
    
    # Extract Celsius temperatures
    celsius_patterns = [
        r'(\d+)\s*°c',
        r'(\d+)\s*degrees?\s*c',
        r'(\d+)\s*c\b'
    ]
    
    # Extract Kelvin temperatures  
    kelvin_patterns = [
        r'(\d+)\s*k\b',
        r'(\d+)\s*kelvin'
    ]
    
    temp_celsius = None
    temp_kelvin = None
    
    # Try to find Celsius temperature
    for pattern in celsius_patterns:
        match = re.search(pattern, note)
        if match:
            temp_celsius = float(match.group(1))
            break
    
    # Try to find Kelvin temperature
    for pattern in kelvin_patterns:
        match = re.search(pattern, note)
        if match:
            temp_kelvin = float(match.group(1))
            break
    
    # Convert Kelvin to Celsius if only Kelvin is found
    if temp_kelvin and not temp_celsius:
        temp_celsius = temp_kelvin - 273.15
    
    # Convert Celsius to Kelvin if only Celsius is found
    if temp_celsius and not temp_kelvin:
        temp_kelvin = temp_celsius + 273.15
    
    return {
        'temperature_celsius': temp_celsius,
        'temperature_kelvin': temp_kelvin
    }

def extract_processing_method(processing_note: str) -> str:
    """Extract processing method from processing notes."""
    if pd.isna(processing_note):
        return 'Unknown'
    
    note = str(processing_note).lower()
    
    # Define processing method patterns
    methods = {
        'arc_melting': ['arc melting', 'arc-melting'],
        'sintering': ['sintering', 'sintered', 'sps', 'spark plasma sintering'],
        'annealing': ['annealing', 'annealed'],
        'pressure_sintering': ['pressure sintered', 'pressure sintering'],
        'as_cast': ['as-cast', 'as cast'],
        'room_temperature': ['room temperature'],
        'cryogenic': ['-196', '-100', '-50', 'cryogenic']
    }
    
    for method, keywords in methods.items():
        for keyword in keywords:
            if keyword in note:
                return method
    
    return 'Other'

def drop_null_features(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop features with high percentage of null values."""
    print(f"\n=== Dropping features with >{threshold*100}% missing values ===")
    
    missing_percentages = (df.isnull().sum() / len(df))
    features_to_drop = missing_percentages[missing_percentages > threshold].index.tolist()
    
    # Don't drop the target variable even if it has missing values
    if 'Compressive strength (MPa)' in features_to_drop:
        features_to_drop.remove('Compressive strength (MPa)')
    
    print(f"Features to drop: {features_to_drop}")
    
    df_cleaned = df.drop(columns=features_to_drop)
    print(f"Shape after dropping features: {df_cleaned.shape}")
    
    return df_cleaned

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features from existing data."""
    print("\n=== Creating derived features ===")
    
    df_enhanced = df.copy()
    
    # Extract temperature and processing method
    temp_info = df['Note on any unqiue processing or data features'].apply(extract_temperature_info)
    df_enhanced['Processing_Temperature_C'] = [info['temperature_celsius'] for info in temp_info]
    df_enhanced['Processing_Temperature_K'] = [info['temperature_kelvin'] for info in temp_info]
    
    df_enhanced['Processing_Method'] = df['Note on any unqiue processing or data features'].apply(extract_processing_method)
    
    # Calculate total composition (should be close to 1.0)
    element_columns = ['Ag', 'Al', 'B', 'C', 'Ca', 'Co', 'Cr', 'Cu', 'Fe', 'Ga', 'Ge', 'Hf', 
                      'Li', 'Mg', 'Mn', 'Mo', 'N', 'Nb', 'Nd', 'Ni', 'Pd', 'Re', 'Sc', 'Si', 
                      'Sn', 'Ta', 'Ti', 'V', 'W', 'Y', 'Zn', 'Zr']
    
    # Only use element columns that exist in the dataframe
    existing_elements = [col for col in element_columns if col in df_enhanced.columns]
    df_enhanced['Total_Composition'] = df_enhanced[existing_elements].sum(axis=1)
    
    # Count number of elements present (non-zero composition)
    df_enhanced['Num_Elements'] = (df_enhanced[existing_elements] > 0).sum(axis=1)
    
    # Calculate Cu and Fe content specifically
    if 'Cu' in df_enhanced.columns:
        df_enhanced['Cu_Content'] = df_enhanced['Cu']
    if 'Fe' in df_enhanced.columns:
        df_enhanced['Fe_Content'] = df_enhanced['Fe']
    
    # Cu to Fe ratio
    if 'Cu' in df_enhanced.columns and 'Fe' in df_enhanced.columns:
        df_enhanced['Cu_Fe_Ratio'] = np.where(df_enhanced['Fe'] > 0, 
                                            df_enhanced['Cu'] / df_enhanced['Fe'], 
                                            df_enhanced['Cu'] * 10)  # Use large value instead of inf
    
    # Phase information - count number of phases
    if 'Phases present' in df_enhanced.columns:
        df_enhanced['Num_Phases'] = df_enhanced['Phases present'].str.count('\\+') + 1
        df_enhanced['Num_Phases'] = df_enhanced['Num_Phases'].fillna(1)
    
    # Crystal structure features
    structure_cols = ['FCC', 'BCC', 'HCP', 'IM']
    existing_structures = [col for col in structure_cols if col in df_enhanced.columns]
    if existing_structures:
        df_enhanced['Num_Crystal_Structures'] = df_enhanced[existing_structures].sum(axis=1)
    
    print(f"Added derived features. New shape: {df_enhanced.shape}")
    return df_enhanced

def clean_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the target variable."""
    print("\n=== Cleaning target variable ===")
    
    # Remove rows where target is missing
    initial_count = len(df)
    df_clean = df.dropna(subset=['Compressive strength (MPa)'])
    final_count = len(df_clean)
    
    print(f"Removed {initial_count - final_count} rows with missing compressive strength")
    print(f"Remaining samples: {final_count}")
    
    # Check for outliers in target variable
    target = df_clean['Compressive strength (MPa)']
    Q1 = target.quantile(0.25)
    Q3 = target.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = ((target < lower_bound) | (target > upper_bound)).sum()
    print(f"Potential outliers in target variable: {outliers}")
    print(f"Target variable range: {target.min():.2f} - {target.max():.2f} MPa")
    print(f"Target variable mean: {target.mean():.2f} ± {target.std():.2f} MPa")
    
    return df_clean

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features."""
    print("\n=== Encoding categorical features ===")
    
    df_encoded = df.copy()
    
    # One-hot encode processing method
    if 'Processing_Method' in df_encoded.columns:
        method_dummies = pd.get_dummies(df_encoded['Processing_Method'], prefix='Method')
        df_encoded = pd.concat([df_encoded, method_dummies], axis=1)
        df_encoded.drop('Processing_Method', axis=1, inplace=True)
    
    # Encode phases present (simplified)
    if 'Phases present' in df_encoded.columns:
        # Create binary features for common phase types
        df_encoded['Has_FCC_Phase'] = df_encoded['Phases present'].str.contains('FCC', na=False).astype(int)
        df_encoded['Has_BCC_Phase'] = df_encoded['Phases present'].str.contains('BCC', na=False).astype(int)
        df_encoded['Has_HCP_Phase'] = df_encoded['Phases present'].str.contains('HCP', na=False).astype(int)
        df_encoded['Has_IM_Phase'] = df_encoded['Phases present'].str.contains('Im|IM', na=False).astype(int)
    
    print(f"Shape after encoding: {df_encoded.shape}")
    return df_encoded

def final_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """Final cleanup and feature selection."""
    print("\n=== Final cleanup ===")
    
    # Drop non-numeric columns that won't be used for modeling
    columns_to_drop = [
        'Id', 'Composition', 'Phases present', 
        'Note on any unqiue processing or data features', 'Source'
    ]
    
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    df_final = df.drop(columns=existing_cols_to_drop)
    
    # Fill remaining missing values with 0 for element compositions
    element_columns = ['Ag', 'Al', 'B', 'C', 'Ca', 'Co', 'Cr', 'Cu', 'Fe', 'Ga', 'Ge', 'Hf', 
                      'Li', 'Mg', 'Mn', 'Mo', 'N', 'Nb', 'Nd', 'Ni', 'Pd', 'Re', 'Sc', 'Si', 
                      'Sn', 'Ta', 'Ti', 'V', 'W', 'Y', 'Zn', 'Zr']
    
    existing_elements = [col for col in element_columns if col in df_final.columns]
    df_final[existing_elements] = df_final[existing_elements].fillna(0)
    
    # Fill other missing values with median for numeric columns
    numeric_columns = df_final.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df_final[col].isnull().sum() > 0:
            median_val = df_final[col].median()
            df_final[col] = df_final[col].fillna(median_val)
            print(f"Filled {col} missing values with median: {median_val}")
    
    # Handle infinite values
    df_final = df_final.replace([np.inf, -np.inf], np.nan)
    
    # Fill any remaining NaN values (from inf replacement) with column median
    for col in numeric_columns:
        if df_final[col].isnull().sum() > 0:
            median_val = df_final[col].median()
            df_final[col] = df_final[col].fillna(median_val)
            print(f"Replaced infinite values in {col} with median: {median_val}")
    
    print(f"Final dataset shape: {df_final.shape}")
    print(f"Final missing values: {df_final.isnull().sum().sum()}")
    print(f"Infinite values: {np.isinf(df_final.select_dtypes(include=[np.number])).sum().sum()}")
    
    return df_final

def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """Save the cleaned dataset."""
    try:
        df.to_csv(output_path, index=False)
        print(f"\nCleaned data saved to: {output_path}")
        
        # Print summary statistics
        print("\n=== Dataset Summary ===")
        print(f"Total samples: {len(df)}")
        print(f"Total features: {len(df.columns)}")
        print(f"Target variable: Compressive strength (MPa)")
        print(f"Target range: {df['Compressive strength (MPa)'].min():.2f} - {df['Compressive strength (MPa)'].max():.2f}")
        
    except Exception as e:
        print(f"Error saving data: {e}")

def main():
    """Main data cleaning pipeline."""
    print("=== Cu-Fe Database Cleaning Pipeline ===")
    
    # Load data
    input_file = "Cu_Fe_DB.csv"
    output_file = "data_processing/Cu_Fe_DB_cleaned.csv"
    
    df = load_data(input_file)
    if df is None:
        return
    
    # Analyze missing data
    analyze_missing_data(df)
    
    # Drop features with high missing values
    df_cleaned = drop_null_features(df, threshold=0.7)  # Drop if >70% missing
    
    # Clean target variable first
    df_cleaned = clean_target_variable(df_cleaned)
    
    # Create derived features
    df_enhanced = create_derived_features(df_cleaned)
    
    # Encode categorical features
    df_encoded = encode_categorical_features(df_enhanced)
    
    # Final cleanup
    df_final = final_cleanup(df_encoded)
    
    # Save cleaned data
    save_cleaned_data(df_final, output_file)
    
    # Print feature importance info
    print("\n=== Key Features for Modeling ===")
    feature_categories = {
        'Target': ['Compressive strength (MPa)'],
        'Mechanical Properties': [col for col in df_final.columns if 'Hardness' in col or 'Strength' in col or 'Elongation' in col],
        'Composition': [col for col in df_final.columns if col in ['Cu', 'Fe', 'Al', 'Ni', 'Co', 'Cr', 'Ti', 'Mn', 'Mo']],
        'Crystal Structure': [col for col in df_final.columns if col in ['FCC', 'BCC', 'HCP', 'IM']],
        'Processing': [col for col in df_final.columns if 'Method_' in col or 'Temperature' in col],
        'Derived Features': [col for col in df_final.columns if any(x in col for x in ['Ratio', 'Num_', 'Total_', 'Has_'])]
    }
    
    for category, features in feature_categories.items():
        existing_features = [f for f in features if f in df_final.columns]
        if existing_features:
            print(f"{category}: {existing_features}")

if __name__ == "__main__":
    main()