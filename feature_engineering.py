#!/usr/bin/env python3
"""
Feature Engineering Script for Cu-Fe HEA ML Prediction
Performs feature engineering on processed data including:
- One-hot encoding of categorical phase features
- Normalization of element composition features
- Generation of derived features and materials properties
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

def load_processed_data(filepath='processed_data.csv'):
    """Load the processed dataset"""
    print(f"Loading processed data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    return df

def encode_phase_features(df):
    """
    Encode categorical phase features using one-hot encoding
    Handles complex phase combinations like 'FCC+BCC', 'FCC+BCC+α', etc.
    """
    print("Encoding phase features...")
    
    # Get unique phases and create binary encoding
    phase_columns = ['FCC', 'BCC', 'HCP', 'IM']
    
    # The data already has binary phase columns, but let's also create additional features
    # from the 'phases' text column for more complex phase combinations
    
    # Create additional phase combination features
    df['phase_FCC_only'] = (df['FCC'] == 1) & (df['BCC'] == 0) & (df['HCP'] == 0) & (df['IM'] == 0)
    df['phase_BCC_only'] = (df['FCC'] == 0) & (df['BCC'] == 1) & (df['HCP'] == 0) & (df['IM'] == 0)
    df['phase_dual_FCC_BCC'] = (df['FCC'] == 1) & (df['BCC'] == 1)
    df['phase_complex'] = (df['FCC'] + df['BCC'] + df['HCP'] + df['IM']) > 2
    
    # Count total number of phases present
    df['total_phases'] = df['FCC'] + df['BCC'] + df['HCP'] + df['IM']
    
    # Convert boolean to int
    bool_cols = ['phase_FCC_only', 'phase_BCC_only', 'phase_dual_FCC_BCC', 'phase_complex']
    for col in bool_cols:
        df[col] = df[col].astype(int)
    
    print(f"Created {len(bool_cols) + 1} additional phase features")
    return df

def normalize_element_features(df):
    """
    Normalize element composition features
    """
    print("Normalizing element composition features...")
    
    # Define element columns (excluding phase columns and other features)
    element_columns = ['Ag', 'Al', 'B', 'C', 'Ca', 'Co', 'Cr', 'Cu', 'Fe', 'Ga', 'Ge', 'Hf', 
                      'Li', 'Mg', 'Mn', 'Mo', 'N', 'Nb', 'Nd', 'Ni', 'Pd', 'Re', 'Sc', 'Si', 
                      'Sn', 'Ta', 'Ti', 'V', 'W', 'Y', 'Zn', 'Zr']
    
    # Filter to only elements that exist in the dataset
    available_elements = [col for col in element_columns if col in df.columns]
    print(f"Found {len(available_elements)} element columns")
    
    # Normalize element compositions using StandardScaler
    scaler = StandardScaler()
    df_normalized = df.copy()
    
    # Apply normalization to element features
    df_normalized[available_elements] = scaler.fit_transform(df[available_elements])
    
    # Store original element fractions for derived feature calculation
    for elem in available_elements:
        df_normalized[f'{elem}_original'] = df[elem]
    
    print("Element composition normalization completed")
    return df_normalized, available_elements

def generate_derived_features(df, element_columns):
    """
    Generate derived features including element ratios and materials properties
    """
    print("Generating derived features...")
    
    # 1. Element ratios (key ratios for Cu-Fe HEAs)
    if 'Cu' in element_columns and 'Fe' in element_columns:
        df['Cu_Fe_ratio'] = df['Cu_original'] / (df['Fe_original'] + 1e-8)  # Avoid division by zero
    
    if 'Al' in element_columns and 'Ni' in element_columns:
        df['Al_Ni_ratio'] = df['Al_original'] / (df['Ni_original'] + 1e-8)
    
    if 'Cr' in element_columns and 'Co' in element_columns:
        df['Cr_Co_ratio'] = df['Cr_original'] / (df['Co_original'] + 1e-8)
    
    # 2. Number of elements present (compositional complexity)
    element_presence = df[[f'{elem}_original' for elem in element_columns]] > 0
    df['num_elements'] = element_presence.sum(axis=1)
    
    # 3. Compositional entropy (mixing entropy approximation)
    element_fractions = df[[f'{elem}_original' for elem in element_columns]]
    # Calculate configurational entropy: S = -R * sum(xi * ln(xi)) where xi > 0
    entropy = 0
    for elem in element_columns:
        xi = df[f'{elem}_original']
        # Only calculate for non-zero fractions
        mask = xi > 0
        entropy_contrib = np.zeros_like(xi)
        entropy_contrib[mask] = xi[mask] * np.log(xi[mask])
        entropy += entropy_contrib
    df['mixing_entropy'] = -entropy  # R constant omitted for simplicity
    
    # 4. Average atomic properties (simplified)
    # Atomic radii approximation (in pm) - simplified values
    atomic_radii = {
        'Al': 143, 'Co': 125, 'Cr': 128, 'Cu': 128, 'Fe': 126, 'Ni': 124,
        'Ti': 147, 'V': 134, 'Mo': 139, 'Mn': 127, 'Zr': 160, 'Nb': 146,
        'Ta': 146, 'W': 139, 'Hf': 159, 'Re': 137
    }
    
    avg_atomic_radius = np.zeros(len(df))
    for elem in element_columns:
        if elem in atomic_radii:
            avg_atomic_radius += df[f'{elem}_original'] * atomic_radii[elem]
    df['avg_atomic_radius'] = avg_atomic_radius
    
    # 5. Atomic size difference (delta parameter)
    delta_sum = np.zeros(len(df))
    for elem in element_columns:
        if elem in atomic_radii:
            delta_sum += df[f'{elem}_original'] * (atomic_radii[elem] - avg_atomic_radius)**2
    df['atomic_size_difference'] = np.sqrt(delta_sum)
    
    # 6. Electronegativity difference (simplified Pauling values)
    electronegativity = {
        'Al': 1.61, 'Co': 1.88, 'Cr': 1.66, 'Cu': 1.90, 'Fe': 1.83, 'Ni': 1.91,
        'Ti': 1.54, 'V': 1.63, 'Mo': 2.16, 'Mn': 1.55, 'Zr': 1.33, 'Nb': 1.6,
        'Ta': 1.5, 'W': 2.36, 'Hf': 1.3, 'Re': 1.9
    }
    
    avg_electronegativity = np.zeros(len(df))
    for elem in element_columns:
        if elem in electronegativity:
            avg_electronegativity += df[f'{elem}_original'] * electronegativity[elem]
    df['avg_electronegativity'] = avg_electronegativity
    
    # 7. Principal element identification
    # Find the element with highest concentration
    element_fractions_only = df[[f'{elem}_original' for elem in element_columns]]
    max_element_idx = element_fractions_only.values.argmax(axis=1)
    df['principal_element'] = [element_columns[idx] for idx in max_element_idx]
    df['principal_element_fraction'] = element_fractions_only.values.max(axis=1)
    
    # 8. Binary, ternary, quaternary classification
    df['alloy_type'] = pd.cut(df['num_elements'], 
                             bins=[0, 2, 3, 4, float('inf')], 
                             labels=['binary', 'ternary', 'quaternary', 'complex'],
                             include_lowest=True)
    
    # One-hot encode alloy type
    alloy_type_encoded = pd.get_dummies(df['alloy_type'], prefix='alloy_type')
    df = pd.concat([df, alloy_type_encoded], axis=1)
    
    # One-hot encode principal element (for most common elements)
    common_elements = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Ni']
    for elem in common_elements:
        if elem in element_columns:
            df[f'principal_{elem}'] = (df['principal_element'] == elem).astype(int)
    
    print(f"Generated {len([col for col in df.columns if col not in element_columns and 'original' not in col])} derived features")
    return df

def create_interaction_features(df, element_columns):
    """
    Create interaction features between key elements
    """
    print("Creating interaction features...")
    
    # Key element pairs for Cu-Fe HEAs
    key_pairs = [('Cu', 'Fe'), ('Al', 'Ni'), ('Cr', 'Co'), ('Ti', 'V')]
    
    for elem1, elem2 in key_pairs:
        if elem1 in element_columns and elem2 in element_columns:
            # Multiplicative interaction
            df[f'{elem1}_{elem2}_interaction'] = df[f'{elem1}_original'] * df[f'{elem2}_original']
            
            # Sum interaction
            df[f'{elem1}_{elem2}_sum'] = df[f'{elem1}_original'] + df[f'{elem2}_original']
    
    print(f"Created interaction features for {len(key_pairs)} element pairs")
    return df

def clean_and_prepare_features(df):
    """
    Clean up the feature set and prepare final dataset
    """
    print("Cleaning and preparing final feature set...")
    
    # Remove original element fraction columns (keep normalized versions)
    original_cols = [col for col in df.columns if col.endswith('_original')]
    df = df.drop(columns=original_cols)
    
    # Remove non-feature columns
    non_feature_cols = ['original_id', 'composition', 'phases', 'processing_notes', 'doi', 
                       'dataset_source', 'principal_element', 'alloy_type']
    cols_to_remove = [col for col in non_feature_cols if col in df.columns]
    df = df.drop(columns=cols_to_remove)
    
    # Handle any remaining missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    print(f"Final feature set contains {len(df.columns)} features")
    return df

def main():
    """Main feature engineering pipeline"""
    print("=== Cu-Fe HEA Feature Engineering Pipeline ===")
    
    # Load processed data
    df = load_processed_data()
    
    # Encode phase features
    df = encode_phase_features(df)
    
    # Normalize element features
    df, element_columns = normalize_element_features(df)
    
    # Generate derived features
    df = generate_derived_features(df, element_columns)
    
    # Create interaction features
    df = create_interaction_features(df, element_columns)
    
    # Clean and prepare final features
    df_features = clean_and_prepare_features(df)
    
    # Save engineered features
    output_file = 'features_data.csv'
    df_features.to_csv(output_file, index=False)
    print(f"\nFeature engineering completed!")
    print(f"Engineered features saved to: {output_file}")
    print(f"Final dataset shape: {df_features.shape}")
    
    # Display feature summary
    print(f"\nFeature Summary:")
    print(f"- Total features: {len(df_features.columns)}")
    print(f"- Samples: {len(df_features)}")
    
    # Check target variable
    if 'compressive_strength' in df_features.columns:
        print(f"- Target variable (compressive_strength) range: {df_features['compressive_strength'].min():.1f} - {df_features['compressive_strength'].max():.1f}")
        print(f"- Target variable mean: {df_features['compressive_strength'].mean():.1f}")
    
    print("\nFeature categories:")
    phase_features = [col for col in df_features.columns if 'phase' in col.lower() or col in ['FCC', 'BCC', 'HCP', 'IM']]
    element_features = [col for col in df_features.columns if any(elem in col for elem in element_columns)]
    derived_features = [col for col in df_features.columns if any(keyword in col.lower() for keyword in ['ratio', 'entropy', 'radius', 'electronegativity', 'interaction', 'sum'])]
    
    print(f"- Phase features: {len(phase_features)}")
    print(f"- Element features: {len(element_features)}")
    print(f"- Derived features: {len(derived_features)}")
    
    return df_features

if __name__ == "__main__":
    engineered_features = main()