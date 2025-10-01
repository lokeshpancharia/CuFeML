"""
Data Utilities for Cu-Fe Database Analysis
Provides helper functions for working with the cleaned dataset including DOI references
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

def load_data_for_modeling(file_path: str = "data_processing/Cu_Fe_DB_cleaned.csv") -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load cleaned data and separate features, target, and source information.
    
    Returns:
        X: Feature matrix (numeric columns only)
        y: Target variable (Compressive strength)
        sources: DOI source information
    """
    df = pd.read_csv(file_path)
    
    # Separate source information
    sources = df['Source'] if 'Source' in df.columns else pd.Series()
    
    # Get target variable
    y = df['Compressive strength (MPa)']
    
    # Get feature matrix (exclude target and source)
    exclude_cols = ['Compressive strength (MPa)', 'Source']
    X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
    
    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target range: {y.min():.1f} - {y.max():.1f} MPa")
    
    return X, y, sources

def get_top_features(n_features: int = 15) -> List[str]:
    """Get the top N features based on previous analysis."""
    top_features = [
        'Hardness (HVN)',
        'BCC',
        'Has_BCC_Phase', 
        'Method_Unknown',
        'Method_arc_melting',
        'Mo',
        'Plasticity (%) - Compressive',
        'Method_sintering',
        'Num_Elements',
        'Num_Phases',
        'Num_Crystal_Structures',
        'Cu_Fe_Ratio',
        'V',
        'Zn',
        'Has_FCC_Phase',
        'Ni',
        'Fe_Content',
        'FCC',
        'Fe',
        'Processing_Temperature_C'
    ]
    
    return top_features[:n_features]

def get_samples_by_source(df: pd.DataFrame, doi_pattern: str) -> pd.DataFrame:
    """
    Filter samples by DOI source pattern.
    
    Args:
        df: Cleaned dataframe
        doi_pattern: Pattern to search in DOI (e.g., '2020.109173' for specific paper)
    
    Returns:
        Filtered dataframe with samples from matching sources
    """
    if 'Source' not in df.columns:
        print("Source column not found in dataframe")
        return pd.DataFrame()
    
    mask = df['Source'].str.contains(doi_pattern, na=False)
    filtered_df = df[mask]
    
    print(f"Found {len(filtered_df)} samples from sources matching '{doi_pattern}'")
    if len(filtered_df) > 0:
        unique_sources = filtered_df['Source'].unique()
        print("Matching sources:")
        for source in unique_sources:
            print(f"  - {source}")
    
    return filtered_df

def analyze_by_source(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze compressive strength statistics by source.
    
    Returns:
        DataFrame with statistics per source
    """
    if 'Source' not in df.columns:
        print("Source column not found in dataframe")
        return pd.DataFrame()
    
    source_stats = df.groupby('Source').agg({
        'Compressive strength (MPa)': ['count', 'mean', 'std', 'min', 'max'],
        'Hardness (HVN)': 'mean',
        'Processing_Temperature_C': 'mean'
    }).round(2)
    
    # Flatten column names
    source_stats.columns = ['_'.join(col).strip() for col in source_stats.columns]
    source_stats = source_stats.reset_index()
    
    # Sort by number of samples
    source_stats = source_stats.sort_values('Compressive strength (MPa)_count', ascending=False)
    
    print("Compressive Strength Statistics by Source:")
    print(source_stats.to_string(index=False))
    
    return source_stats

def get_processing_method_by_source(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze processing methods used by each source.
    
    Returns:
        DataFrame showing processing method distribution by source
    """
    if 'Source' not in df.columns:
        print("Source column not found in dataframe")
        return pd.DataFrame()
    
    # Get method columns
    method_cols = [col for col in df.columns if col.startswith('Method_')]
    
    if not method_cols:
        print("No processing method columns found")
        return pd.DataFrame()
    
    # Create summary by source
    source_methods = []
    
    for source in df['Source'].unique():
        source_data = df[df['Source'] == source]
        methods_used = []
        
        for method_col in method_cols:
            if source_data[method_col].sum() > 0:
                method_name = method_col.replace('Method_', '')
                count = source_data[method_col].sum()
                methods_used.append(f"{method_name}({count})")
        
        source_methods.append({
            'Source': source,
            'Sample_Count': len(source_data),
            'Methods_Used': ', '.join(methods_used) if methods_used else 'None',
            'Avg_Compressive_Strength': source_data['Compressive strength (MPa)'].mean()
        })
    
    result_df = pd.DataFrame(source_methods)
    result_df = result_df.sort_values('Sample_Count', ascending=False)
    
    print("Processing Methods by Source:")
    print(result_df.to_string(index=False))
    
    return result_df

def create_source_reference_guide(df: pd.DataFrame) -> None:
    """
    Create a reference guide mapping compositions to their sources.
    """
    if 'Source' not in df.columns:
        print("Source column not found in dataframe")
        return
    
    # Load original data to get composition names
    try:
        original_df = pd.read_csv("Cu_Fe_DB.csv")
        if 'Composition' in original_df.columns and 'Source' in original_df.columns:
            reference_guide = original_df[['Composition', 'Source', 'Compressive strength (MPa)']].copy()
            reference_guide = reference_guide.sort_values('Compressive strength (MPa)', ascending=False)
            
            # Save reference guide
            reference_guide.to_csv("data_processing/source_reference_guide.csv", index=False)
            print("Source reference guide saved to: data_processing/source_reference_guide.csv")
            
            # Print summary
            print(f"\nSource Reference Summary:")
            print(f"Total unique sources: {reference_guide['Source'].nunique()}")
            print(f"Total compositions: {len(reference_guide)}")
            
            # Top sources by sample count
            source_counts = reference_guide['Source'].value_counts().head(5)
            print(f"\nTop 5 sources by sample count:")
            for source, count in source_counts.items():
                print(f"  {count} samples: {source}")
                
        else:
            print("Could not find Composition and Source columns in original data")
            
    except FileNotFoundError:
        print("Original Cu_Fe_DB.csv file not found")

def main():
    """Demonstrate utility functions."""
    print("=== Cu-Fe Database Utilities Demo ===")
    
    # Load data
    X, y, sources = load_data_for_modeling()
    df = pd.read_csv("data_processing/Cu_Fe_DB_cleaned.csv")
    
    print(f"\n=== Source Analysis ===")
    
    # Analyze by source
    source_stats = analyze_by_source(df)
    
    print(f"\n=== Processing Methods by Source ===")
    method_analysis = get_processing_method_by_source(df)
    
    print(f"\n=== Creating Reference Guide ===")
    create_source_reference_guide(df)
    
    print(f"\n=== Example: Filter by specific source ===")
    # Example: Get samples from a specific paper
    vacuum_samples = get_samples_by_source(df, "vacuum.2020.109173")
    if len(vacuum_samples) > 0:
        print(f"Compressive strength range: {vacuum_samples['Compressive strength (MPa)'].min():.0f} - {vacuum_samples['Compressive strength (MPa)'].max():.0f} MPa")
    
    print(f"\n=== Top Features for Modeling ===")
    top_features = get_top_features(10)
    print("Recommended features:", top_features)

if __name__ == "__main__":
    main()