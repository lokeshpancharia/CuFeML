"""
Data Cleaning Script for New Database (database (1).csv)
Target: Compressive Strength Prediction and Merging with Existing Database

This script:
1. Cleans the new database file
2. Filters out tensile test data (Test Type = 'T')
3. Extracts composition information from HTML-formatted strings
4. Prepares data for merging with existing Cu_Fe_DB_cleaned.csv
5. Handles duplicate detection and removal
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

def load_new_database(file_path: str) -> pd.DataFrame:
    """Load the new database CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"New database loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading new database: {e}")
        return None

def analyze_new_database(df: pd.DataFrame) -> None:
    """Analyze the structure and content of the new database."""
    print("\n=== New Database Analysis ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check missing values
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    print(f"\nMissing Data:")
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  {col}: {count} ({missing_percentages[col]:.1f}%)")
    
    # Check Test Type distribution
    if 'Test Type' in df.columns:
        test_types = df['Test Type'].value_counts(dropna=False)
        print(f"\nTest Type Distribution:")
        for test_type, count in test_types.items():
            print(f"  {test_type}: {count}")
    
    # Check target variable (compressive strength)
    target_col = 'σ (MPa) at 23 C'
    if target_col in df.columns:
        target_stats = df[target_col].describe()
        print(f"\nTarget Variable ({target_col}) Statistics:")
        print(target_stats)

def filter_compressive_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out tensile test data, keep only compressive strength data."""
    print("\n=== Filtering Compressive Data ===")
    
    initial_count = len(df)
    
    # Remove rows where Test Type is 'T' (tensile)
    if 'Test Type' in df.columns:
        df_filtered = df[df['Test Type'] != 'T'].copy()
        tensile_removed = initial_count - len(df_filtered)
        print(f"Removed {tensile_removed} tensile test samples")
    else:
        df_filtered = df.copy()
        print("No Test Type column found")
    
    # Remove rows where compressive strength is missing
    target_col = 'σ (MPa) at 23 C'
    if target_col in df_filtered.columns:
        df_filtered = df_filtered.dropna(subset=[target_col])
        missing_target_removed = len(df) - tensile_removed - len(df_filtered)
        print(f"Removed {missing_target_removed} samples with missing compressive strength")
    
    print(f"Remaining samples: {len(df_filtered)} (from {initial_count})")
    
    return df_filtered

def parse_composition_html(composition_str: str) -> Dict[str, float]:
    """
    Parse HTML-formatted composition strings to extract element fractions.
    
    Examples:
    - 'CoCrCuFeNi' -> equal fractions for each element
    - 'Al<sub>0.5</sub>CoCrCuFeNi' -> Al: 0.5, others equal
    - 'Al<sub>22.5</sub>Cu<sub>20</sub>Fe<sub>15</sub>Ni<sub>20</sub>Ti<sub>22.5</sub>' -> specific fractions
    """
    if pd.isna(composition_str):
        return {}
    
    composition_str = str(composition_str)
    
    # Dictionary to store element fractions
    elements = {}
    
    # Pattern to match element with subscript: Element<sub>fraction</sub>
    pattern_with_sub = r'([A-Z][a-z]?)<sub>([0-9.]+)</sub>'
    
    # Find all elements with explicit fractions
    matches_with_sub = re.findall(pattern_with_sub, composition_str)
    
    for element, fraction in matches_with_sub:
        elements[element] = float(fraction)
    
    # Remove the matched parts to find elements without explicit fractions
    remaining_str = re.sub(pattern_with_sub, '', composition_str)
    
    # Pattern to match standalone elements
    pattern_standalone = r'([A-Z][a-z]?)(?![a-z])'
    standalone_elements = re.findall(pattern_standalone, remaining_str)
    
    # If there are standalone elements, assume equal fractions
    if standalone_elements:
        # Calculate remaining fraction for standalone elements
        used_fraction = sum(elements.values())
        remaining_fraction = max(0, 100 - used_fraction)  # Assume percentages
        
        if remaining_fraction > 0 and len(standalone_elements) > 0:
            equal_fraction = remaining_fraction / len(standalone_elements)
            for element in standalone_elements:
                elements[element] = equal_fraction
        elif len(standalone_elements) > 0 and not elements:
            # All elements are standalone, assume equal fractions
            equal_fraction = 100 / len(standalone_elements)
            for element in standalone_elements:
                elements[element] = equal_fraction
    
    # Convert to atomic fractions (0-1 scale)
    total = sum(elements.values())
    if total > 0:
        elements = {k: v/total for k, v in elements.items()}
    
    return elements

def create_composition_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create composition features from the Composition column."""
    print("\n=== Creating Composition Features ===")
    
    df_enhanced = df.copy()
    
    # Parse all compositions
    all_elements = set()
    compositions = []
    
    for comp_str in df['Composition']:
        comp_dict = parse_composition_html(comp_str)
        compositions.append(comp_dict)
        all_elements.update(comp_dict.keys())
    
    print(f"Found elements: {sorted(all_elements)}")
    
    # Create columns for each element
    for element in sorted(all_elements):
        df_enhanced[element] = [comp.get(element, 0.0) for comp in compositions]
    
    # Create derived features
    df_enhanced['Total_Composition'] = df_enhanced[sorted(all_elements)].sum(axis=1)
    df_enhanced['Num_Elements'] = (df_enhanced[sorted(all_elements)] > 0).sum(axis=1)
    
    # Cu and Fe specific features if present
    if 'Cu' in all_elements and 'Fe' in all_elements:
        df_enhanced['Cu_Fe_Ratio'] = np.where(df_enhanced['Fe'] > 0, 
                                            df_enhanced['Cu'] / df_enhanced['Fe'], 
                                            df_enhanced['Cu'] * 10)
    
    print(f"Created {len(all_elements)} element composition features")
    
    return df_enhanced, sorted(all_elements)

def process_phase_information(df: pd.DataFrame) -> pd.DataFrame:
    """Process phase information to create binary features."""
    print("\n=== Processing Phase Information ===")
    
    df_enhanced = df.copy()
    
    if 'Phase' in df.columns:
        # Create binary features for common phases
        df_enhanced['Has_FCC_Phase'] = df['Phase'].str.contains('FCC', na=False).astype(int)
        df_enhanced['Has_BCC_Phase'] = df['Phase'].str.contains('BCC', na=False).astype(int)
        df_enhanced['Has_HCP_Phase'] = df['Phase'].str.contains('HCP', na=False).astype(int)
        df_enhanced['Has_IM_Phase'] = df['Phase'].str.contains('Im|IM', na=False).astype(int)
        
        # Count number of phases
        df_enhanced['Num_Phases'] = df['Phase'].str.count('\\+') + 1
        df_enhanced['Num_Phases'] = df_enhanced['Num_Phases'].fillna(1)
        
        # Create crystal structure indicators (for compatibility with old database)
        df_enhanced['FCC'] = df_enhanced['Has_FCC_Phase']
        df_enhanced['BCC'] = df_enhanced['Has_BCC_Phase'] 
        df_enhanced['HCP'] = df_enhanced['Has_HCP_Phase']
        df_enhanced['IM'] = df_enhanced['Has_IM_Phase']
        
        print("Created phase-based features")
    
    return df_enhanced

def clean_doi_column(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize DOI column."""
    print("\n=== Cleaning DOI Information ===")
    
    df_enhanced = df.copy()
    
    if 'doi' in df.columns:
        # Standardize DOI format
        df_enhanced['Source'] = df['doi'].apply(lambda x: f"https://{x}" if pd.notna(x) and not str(x).startswith('http') else x)
        print("Standardized DOI format in Source column")
    
    return df_enhanced

def rename_and_standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to match the existing database format."""
    print("\n=== Standardizing Column Names ===")
    
    # Column mapping from new database to old database format
    column_mapping = {
        'σ (MPa) at 23 C': 'Compressive strength (MPa)',
        'ρ (g/cm<sup>3</sup>)': 'Density (g/cm3)',
        'E (GPa)': 'Elastic Modulus (GPa)',
        'G (GPa)': 'Shear Modulus (GPa)',
        'C11 (GPa)': 'C11 Elastic Constant (GPa)',
        'Fracture Toughness': 'Fracture Toughness',
        'Ductility [%]': 'Ductility (%)',
        'Phase': 'Phases present'
    }
    
    df_renamed = df.rename(columns=column_mapping)
    
    # Add missing columns that exist in the old database (fill with appropriate defaults)
    missing_cols = {
        'Hardness (HVN)': np.nan,
        'Plasticity (%) - Compressive': np.nan,
        'Processing_Temperature_C': np.nan,
        'Processing_Temperature_K': np.nan,
        'Num_Crystal_Structures': 0
    }
    
    for col, default_val in missing_cols.items():
        if col not in df_renamed.columns:
            df_renamed[col] = default_val
    
    print(f"Renamed columns and added missing columns")
    
    return df_renamed

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    print("\n=== Handling Missing Values ===")
    
    df_clean = df.copy()
    
    # Fill missing values for numeric columns
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if df_clean[col].isnull().sum() > 0:
            if col in ['Hardness (HVN)', 'Plasticity (%) - Compressive', 'Processing_Temperature_C', 'Processing_Temperature_K']:
                # Use median from existing database for these columns
                median_values = {
                    'Hardness (HVN)': 410.0,
                    'Plasticity (%) - Compressive': 11.3,
                    'Processing_Temperature_C': 499.85,
                    'Processing_Temperature_K': 773.0
                }
                fill_value = median_values.get(col, df_clean[col].median())
            else:
                fill_value = df_clean[col].median()
                if pd.isna(fill_value):  # If median is also NaN (all values missing)
                    fill_value = 0.0
            
            df_clean[col] = df_clean[col].fillna(fill_value)
            print(f"Filled {col} missing values with {fill_value}")
    
    # Handle infinite values
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    # Fill any remaining NaN values
    for col in numeric_columns:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
    
    print(f"Final missing values: {df_clean.isnull().sum().sum()}")
    
    return df_clean

def prepare_for_merging(df: pd.DataFrame, element_list: List[str]) -> pd.DataFrame:
    """Prepare the new database for merging with the existing database."""
    print("\n=== Preparing for Merging ===")
    
    # Load existing database to get column structure
    try:
        existing_df = pd.read_csv("data_processing/Cu_Fe_DB_cleaned.csv")
        existing_columns = set(existing_df.columns)
        print(f"Existing database has {len(existing_columns)} columns")
    except FileNotFoundError:
        print("Warning: Could not load existing database for column matching")
        existing_columns = set()
    
    df_prepared = df.copy()
    
    # Ensure all elements from existing database are present
    all_possible_elements = ['Ag', 'Al', 'B', 'C', 'Ca', 'Co', 'Cr', 'Cu', 'Fe', 'Ga', 'Ge', 'Hf', 
                           'Li', 'Mg', 'Mn', 'Mo', 'N', 'Nb', 'Nd', 'Ni', 'Pd', 'Re', 'Sc', 'Si', 
                           'Sn', 'Ta', 'Ti', 'V', 'W', 'Y', 'Zn', 'Zr']
    
    for element in all_possible_elements:
        if element not in df_prepared.columns:
            df_prepared[element] = 0.0
    
    # Add missing method columns (all False for new database)
    method_columns = ['Method_Other', 'Method_Unknown', 'Method_annealing', 'Method_arc_melting', 
                     'Method_as_cast', 'Method_room_temperature', 'Method_sintering']
    
    for method_col in method_columns:
        if method_col not in df_prepared.columns:
            df_prepared[method_col] = False
    
    # Set Method_Unknown to True for all new samples (since we don't have processing info)
    df_prepared['Method_Unknown'] = True
    
    # Add content features
    if 'Cu' in df_prepared.columns:
        df_prepared['Cu_Content'] = df_prepared['Cu']
    if 'Fe' in df_prepared.columns:
        df_prepared['Fe_Content'] = df_prepared['Fe']
    
    # Calculate Num_Crystal_Structures
    structure_cols = ['FCC', 'BCC', 'HCP', 'IM']
    existing_structures = [col for col in structure_cols if col in df_prepared.columns]
    if existing_structures:
        df_prepared['Num_Crystal_Structures'] = df_prepared[existing_structures].sum(axis=1)
    
    print(f"Prepared dataset shape: {df_prepared.shape}")
    
    return df_prepared

def detect_duplicates_with_existing(new_df: pd.DataFrame, existing_path: str = "data_processing/Cu_Fe_DB_cleaned.csv") -> pd.DataFrame:
    """Detect and handle duplicates between new and existing databases."""
    print("\n=== Detecting Duplicates with Existing Database ===")
    
    try:
        existing_df = pd.read_csv(existing_path)
        print(f"Loaded existing database: {existing_df.shape}")
        
        # Compare based on composition and compressive strength (allowing small tolerance)
        duplicates_mask = []
        
        element_cols = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Mo', 'Ni', 'Ti', 'Si', 'V', 'Zn', 'Zr']
        existing_elements = [col for col in element_cols if col in existing_df.columns and col in new_df.columns]
        
        for idx, new_row in new_df.iterrows():
            is_duplicate = False
            
            for _, existing_row in existing_df.iterrows():
                # Check composition similarity (within 0.01 tolerance)
                comp_similar = True
                for element in existing_elements:
                    if abs(new_row[element] - existing_row[element]) > 0.01:
                        comp_similar = False
                        break
                
                # Check compressive strength similarity (within 5% tolerance)
                new_strength = new_row['Compressive strength (MPa)']
                existing_strength = existing_row['Compressive strength (MPa)']
                
                if comp_similar and abs(new_strength - existing_strength) / max(new_strength, existing_strength) < 0.05:
                    is_duplicate = True
                    break
            
            duplicates_mask.append(is_duplicate)
        
        # Remove duplicates
        duplicates_series = pd.Series(duplicates_mask, index=new_df.index)
        new_df_clean = new_df[~duplicates_series].copy()
        duplicates_removed = sum(duplicates_mask)
        
        print(f"Removed {duplicates_removed} duplicate samples")
        print(f"Remaining new samples: {len(new_df_clean)}")
        
        return new_df_clean
        
    except FileNotFoundError:
        print("Existing database not found, skipping duplicate detection")
        return new_df

def save_cleaned_new_database(df: pd.DataFrame, output_path: str) -> None:
    """Save the cleaned new database."""
    try:
        df.to_csv(output_path, index=False)
        print(f"\nCleaned new database saved to: {output_path}")
        
        # Print summary statistics
        print("\n=== New Database Summary ===")
        print(f"Total samples: {len(df)}")
        print(f"Total features: {len(df.columns)}")
        print(f"Target variable: Compressive strength (MPa)")
        if 'Compressive strength (MPa)' in df.columns:
            target = df['Compressive strength (MPa)']
            print(f"Target range: {target.min():.2f} - {target.max():.2f} MPa")
            print(f"Target mean: {target.mean():.2f} ± {target.std():.2f} MPa")
        
    except Exception as e:
        print(f"Error saving cleaned new database: {e}")

def merge_databases(new_db_path: str, existing_db_path: str, output_path: str) -> None:
    """Merge the cleaned new database with the existing database."""
    print("\n=== Merging Databases ===")
    
    try:
        # Load both databases
        new_df = pd.read_csv(new_db_path)
        existing_df = pd.read_csv(existing_db_path)
        
        print(f"New database: {new_df.shape}")
        print(f"Existing database: {existing_df.shape}")
        
        # Ensure column compatibility
        all_columns = set(new_df.columns) | set(existing_df.columns)
        
        # Add missing columns to both dataframes
        for col in all_columns:
            if col not in new_df.columns:
                if col in ['Method_Unknown']:
                    new_df[col] = True
                elif col.startswith('Method_'):
                    new_df[col] = False
                else:
                    new_df[col] = 0 if new_df.select_dtypes(include=[np.number]).columns.tolist() else ''
            
            if col not in existing_df.columns:
                if col in ['Method_Unknown']:
                    existing_df[col] = True
                elif col.startswith('Method_'):
                    existing_df[col] = False
                else:
                    existing_df[col] = 0 if existing_df.select_dtypes(include=[np.number]).columns.tolist() else ''
        
        # Reorder columns to match
        column_order = sorted(all_columns)
        new_df = new_df[column_order]
        existing_df = existing_df[column_order]
        
        # Merge databases
        merged_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Final duplicate check within merged dataset
        initial_count = len(merged_df)
        merged_df = merged_df.drop_duplicates()
        final_count = len(merged_df)
        
        print(f"Removed {initial_count - final_count} internal duplicates")
        
        # Save merged database
        merged_df.to_csv(output_path, index=False)
        
        print(f"\nMerged database saved to: {output_path}")
        print(f"Final merged database shape: {merged_df.shape}")
        
        # Summary statistics
        if 'Compressive strength (MPa)' in merged_df.columns:
            target = merged_df['Compressive strength (MPa)']
            print(f"Merged target range: {target.min():.2f} - {target.max():.2f} MPa")
            print(f"Merged target mean: {target.mean():.2f} ± {target.std():.2f} MPa")
        
    except Exception as e:
        print(f"Error merging databases: {e}")

def main():
    """Main cleaning and merging pipeline."""
    print("=== New Database Cleaning and Merging Pipeline ===")
    
    # File paths
    input_file = "database (1).csv"
    cleaned_new_output = "data_processing/new_database_cleaned.csv"
    existing_db_path = "data_processing/Cu_Fe_DB_cleaned.csv"
    merged_output = "data_processing/merged_database_cleaned.csv"
    
    # Load and analyze new database
    df = load_new_database(input_file)
    if df is None:
        return
    
    analyze_new_database(df)
    
    # Filter compressive data (remove tensile tests)
    df_filtered = filter_compressive_data(df)
    
    # Create composition features
    df_enhanced, element_list = create_composition_features(df_filtered)
    
    # Process phase information
    df_enhanced = process_phase_information(df_enhanced)
    
    # Clean DOI information
    df_enhanced = clean_doi_column(df_enhanced)
    
    # Rename and standardize columns
    df_standardized = rename_and_standardize_columns(df_enhanced)
    
    # Handle missing values
    df_clean = handle_missing_values(df_standardized)
    
    # Prepare for merging
    df_prepared = prepare_for_merging(df_clean, element_list)
    
    # Detect and remove duplicates with existing database
    df_no_duplicates = detect_duplicates_with_existing(df_prepared, existing_db_path)
    
    # Save cleaned new database
    save_cleaned_new_database(df_no_duplicates, cleaned_new_output)
    
    # Merge with existing database
    merge_databases(cleaned_new_output, existing_db_path, merged_output)
    
    print("\n=== Pipeline Complete ===")
    print(f"Cleaned new database: {cleaned_new_output}")
    print(f"Merged database: {merged_output}")

if __name__ == "__main__":
    main()