#!/usr/bin/env python3
"""
Data Loader Script for Cu-Fe HEA ML Prediction
Loads, merges, and cleans CSV datasets with DOI-based duplicate detection
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Handles loading, merging, and cleaning of HEA datasets"""
    
    def __init__(self):
        self.raw_data = {}
        self.merged_data = None
        self.merge_report = {}
        
    def load_datasets(self, file_paths: List[str]) -> Dict[str, pd.DataFrame]:
        """Load CSV files and return dictionary of dataframes"""
        datasets = {}
        
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path)
                datasets[file_path] = df
                print(f"Loaded {file_path}: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        self.raw_data = datasets
        return datasets
    
    def standardize_doi(self, doi: str) -> str:
        """Standardize DOI format for comparison"""
        if pd.isna(doi) or doi == '':
            return ''
        
        # Convert to string and clean
        doi_str = str(doi).strip().lower()
        
        # Remove common prefixes
        doi_str = re.sub(r'^(https?://)?(dx\.)?doi\.org/', '', doi_str)
        doi_str = re.sub(r'^doi:', '', doi_str)
        
        return doi_str
    
    def extract_composition_elements(self, composition: str) -> Dict[str, float]:
        """Extract element fractions from composition string"""
        if pd.isna(composition):
            return {}
            
        elements = {}
        # Handle subscript notation like Al<sub>0.5</sub>CoCrCuFeNi
        composition = re.sub(r'<sub>([^<]+)</sub>', r'\1', composition)
        
        # Pattern to match element and optional coefficient
        pattern = r'([A-Z][a-z]?)([0-9]*\.?[0-9]*)'
        matches = re.findall(pattern, composition)
        
        total_fraction = 0
        for element, fraction in matches:
            if fraction == '':
                fraction = 1.0
            else:
                fraction = float(fraction)
            elements[element] = fraction
            total_fraction += fraction
        
        # Normalize to fractions
        if total_fraction > 0:
            for element in elements:
                elements[element] = elements[element] / total_fraction
                
        return elements
    
    def detect_duplicates_by_doi_and_composition(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """Detect duplicates based on DOI AND composition matching"""
        duplicate_groups = {}
        
        for idx, row in df.iterrows():
            doi = self.standardize_doi(row.get('Source', row.get('doi', '')))
            composition = str(row.get('Composition', '')).strip()
            
            if doi and doi != '' and composition and composition != '':
                # Create a key combining DOI and composition
                key = f"{doi}||{composition}"
                
                if key not in duplicate_groups:
                    duplicate_groups[key] = []
                duplicate_groups[key].append(idx)
        
        # Only return groups with multiple entries (true duplicates)
        duplicates = {key: indices for key, indices in duplicate_groups.items() if len(indices) > 1}
        
        return duplicates
    
    def detect_duplicates_by_composition(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """Detect duplicates based on composition similarity"""
        composition_groups = {}
        
        for idx, row in df.iterrows():
            composition = row.get('Composition', '')
            if pd.isna(composition):
                continue
                
            # Normalize composition string
            comp_normalized = re.sub(r'[^A-Za-z0-9.]', '', str(composition)).lower()
            
            if comp_normalized:
                if comp_normalized not in composition_groups:
                    composition_groups[comp_normalized] = []
                composition_groups[comp_normalized].append(idx)
        
        # Only return groups with multiple entries
        duplicates = {comp: indices for comp, indices in composition_groups.items() if len(indices) > 1}
        
        return duplicates 
   
    def merge_duplicate_records(self, df: pd.DataFrame, duplicate_groups: Dict[str, List[int]]) -> pd.DataFrame:
        """Merge duplicate records to maximize data retention"""
        merged_df = df.copy()
        rows_to_drop = []
        
        for group_key, indices in duplicate_groups.items():
            if len(indices) <= 1:
                continue
                
            # Get all rows in this duplicate group
            group_rows = df.loc[indices]
            
            # Create merged row starting with first row
            merged_row = group_rows.iloc[0].copy()
            
            # Merge data from all rows, prioritizing non-null values
            for col in group_rows.columns:
                values = group_rows[col].dropna()
                if len(values) > 0:
                    if col in ['Compressive strength (MPa)', 'σ (MPa) at 23 C']:
                        # For target variable, take mean of available values
                        merged_row[col] = values.mean()
                    elif col in ['Hardness (HVN)', 'Yield Strength (MPa)', 'Ultimate Tensile Strength (MPa)', 
                               'Elongation (%)', 'Plasticity (%) - Compressive']:
                        # For other numeric properties, take mean
                        numeric_values = pd.to_numeric(values, errors='coerce').dropna()
                        if len(numeric_values) > 0:
                            merged_row[col] = numeric_values.mean()
                    elif col in ['Phases present', 'Phase']:
                        # For phases, combine unique values
                        unique_phases = set()
                        for phase_str in values:
                            if pd.notna(phase_str):
                                unique_phases.update([p.strip() for p in str(phase_str).split('+')])
                        merged_row[col] = '+'.join(sorted(unique_phases)) if unique_phases else values.iloc[0]
                    else:
                        # For other columns, take first non-null value
                        merged_row[col] = values.iloc[0]
            
            # Update the first row with merged data
            merged_df.loc[indices[0]] = merged_row
            
            # Mark other rows for deletion
            rows_to_drop.extend(indices[1:])
        
        # Drop duplicate rows
        merged_df = merged_df.drop(rows_to_drop)
        
        return merged_df
    
    def standardize_column_names(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Standardize column names across datasets"""
        df_std = df.copy()
        
        # Column mapping for standardization
        column_mapping = {
            # Compressive strength target variable
            'Compressive strength (MPa)': 'compressive_strength',
            'σ (MPa) at 23 C': 'compressive_strength',
            
            # Basic properties
            'Composition': 'composition',
            'Phases present': 'phases',
            'Phase': 'phases',
            'Hardness (HVN)': 'hardness',
            'Yield Strength (MPa)': 'yield_strength',
            'Ultimate Tensile Strength (MPa)': 'ultimate_tensile_strength',
            'Elongation (%)': 'elongation',
            'Plasticity (%) - Compressive': 'plasticity',
            'Ductility [%]': 'ductility',
            
            # Source information
            'Source': 'doi',
            'doi': 'doi',
            
            # Processing notes
            'Note on any unqiue processing or data features': 'processing_notes',
            
            # ID columns
            'Id': 'original_id',
            'Unique ID': 'original_id'
        }
        
        # Rename columns
        df_std = df_std.rename(columns=column_mapping)
        
        # Add dataset source
        df_std['dataset_source'] = dataset_name
        
        return df_std
    
    def merge_datasets(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple datasets with duplicate detection and resolution"""
        all_dataframes = []
        
        # Standardize each dataset
        for dataset_name, df in datasets.items():
            df_std = self.standardize_column_names(df, dataset_name)
            all_dataframes.append(df_std)
        
        # Combine all datasets
        combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
        
        print(f"Combined datasets: {combined_df.shape[0]} total rows")
        
        # Detect duplicates by DOI AND composition (true duplicates only)
        doi_comp_duplicates = self.detect_duplicates_by_doi_and_composition(combined_df)
        print(f"Found {len(doi_comp_duplicates)} DOI+composition duplicate groups")
        
        # Merge only true duplicates (same DOI AND same composition)
        if doi_comp_duplicates:
            combined_df = self.merge_duplicate_records(combined_df, doi_comp_duplicates)
            print(f"After DOI+composition merge: {combined_df.shape[0]} rows")
        
        # Optionally detect composition-only duplicates for records without DOI
        no_doi_df = combined_df[combined_df['doi'] == '']
        if len(no_doi_df) > 0:
            comp_duplicates = self.detect_duplicates_by_composition(no_doi_df)
            if comp_duplicates:
                combined_df = self.merge_duplicate_records(combined_df, comp_duplicates)
                print(f"After composition-only merge: {combined_df.shape[0]} rows")
        
        self.merged_data = combined_df
        self.merge_report = {
            'original_total_rows': sum(df.shape[0] for df in datasets.values()),
            'final_rows': combined_df.shape[0],
            'doi_composition_duplicate_groups': len(doi_comp_duplicates),
            'rows_merged': sum(df.shape[0] for df in datasets.values()) - combined_df.shape[0]
        }
        
        return combined_df    

    def handle_missing_compressive_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in compressive strength target variable"""
        df_clean = df.copy()
        
        # Count missing values
        missing_count = df_clean['compressive_strength'].isna().sum()
        total_count = len(df_clean)
        
        print(f"Missing compressive strength values: {missing_count}/{total_count} ({missing_count/total_count*100:.1f}%)")
        
        # Remove rows with missing compressive strength (target variable)
        df_clean = df_clean.dropna(subset=['compressive_strength'])
        
        print(f"Rows after removing missing target values: {len(df_clean)}")
        
        return df_clean
    
    def extract_element_fractions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract element fractions from composition strings"""
        df_with_elements = df.copy()
        
        # Get all unique elements from both datasets
        all_elements = set()
        
        # Check if element columns already exist (from Cu_Fe_DB.csv)
        element_columns = [col for col in df.columns if len(col) <= 2 and col.isalpha() and col[0].isupper()]
        
        if element_columns:
            # Use existing element columns
            all_elements.update(element_columns)
        else:
            # Extract elements from composition strings
            for composition in df['composition'].dropna():
                elements = self.extract_composition_elements(composition)
                all_elements.update(elements.keys())
        
        # Ensure we have the key elements for Cu-Fe HEAs
        key_elements = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Ni', 'Ti', 'Mn', 'V', 'Mo', 'Zr', 'Hf', 'Y']
        all_elements.update(key_elements)
        
        # Initialize element fraction columns
        for element in sorted(all_elements):
            if element not in df_with_elements.columns:
                df_with_elements[element] = 0.0
        
        # Fill element fractions
        for idx, row in df_with_elements.iterrows():
            # If element columns already exist and have values, use them
            if element_columns and not all(pd.isna(row[col]) or row[col] == 0 for col in element_columns):
                continue
                
            # Otherwise, extract from composition string
            composition = row['composition']
            if pd.notna(composition):
                elements = self.extract_composition_elements(composition)
                for element, fraction in elements.items():
                    if element in df_with_elements.columns:
                        df_with_elements.at[idx, element] = fraction
        
        return df_with_elements
    
    def clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final data cleaning and validation"""
        df_clean = df.copy()
        
        # Remove rows with invalid compressive strength values
        df_clean = df_clean[df_clean['compressive_strength'] > 0]
        
        # Ensure composition is not empty
        df_clean = df_clean[df_clean['composition'].notna()]
        df_clean = df_clean[df_clean['composition'] != '']
        
        # Fill missing phases with 'Unknown'
        df_clean['phases'] = df_clean['phases'].fillna('Unknown')
        
        # Fill missing DOI with empty string
        df_clean['doi'] = df_clean['doi'].fillna('')
        
        # Convert numeric columns
        numeric_columns = ['compressive_strength', 'hardness', 'yield_strength', 
                          'ultimate_tensile_strength', 'elongation', 'plasticity']
        
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        print(f"Final cleaned dataset: {len(df_clean)} rows")
        
        return df_clean
    
    def detect_potential_duplicates_by_doi(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """Detect potential duplicates by DOI only (for reporting)"""
        doi_groups = {}
        
        for idx, row in df.iterrows():
            doi = self.standardize_doi(row.get('Source', row.get('doi', '')))
            
            if doi and doi != '':
                if doi not in doi_groups:
                    doi_groups[doi] = []
                doi_groups[doi].append(idx)
        
        # Only return groups with multiple entries
        duplicates = {doi: indices for doi, indices in doi_groups.items() if len(indices) > 1}
        
        return duplicates
    
    def generate_merge_report(self) -> Dict:
        """Generate detailed merge report"""
        if not self.merge_report:
            return {}
            
        report = self.merge_report.copy()
        
        if self.merged_data is not None:
            # Add data quality metrics
            report['compressive_strength_coverage'] = (
                self.merged_data['compressive_strength'].notna().sum() / len(self.merged_data)
            )
            
            report['unique_compositions'] = self.merged_data['composition'].nunique()
            report['unique_dois'] = self.merged_data['doi'].nunique()
            
            # Phase distribution
            phase_counts = self.merged_data['phases'].value_counts().head(5).to_dict()
            report['top_phases'] = phase_counts
            
            # Report potential duplicates by DOI (not merged)
            potential_doi_duplicates = self.detect_potential_duplicates_by_doi(self.merged_data)
            report['potential_doi_duplicate_groups'] = len(potential_doi_duplicates)
            report['potential_doi_duplicate_records'] = sum(len(indices) for indices in potential_doi_duplicates.values())
        
        return report
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str = 'processed_data.csv'):
        """Save processed data to CSV"""
        df.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")
        
        # Print summary statistics
        print(f"\nDataset Summary:")
        print(f"Total samples: {len(df)}")
        print(f"Compressive strength range: {df['compressive_strength'].min():.1f} - {df['compressive_strength'].max():.1f} MPa")
        print(f"Mean compressive strength: {df['compressive_strength'].mean():.1f} MPa")
        print(f"Unique compositions: {df['composition'].nunique()}")
        print(f"Phase distribution:")
        print(df['phases'].value_counts().head())


def main():
    """Main execution function"""
    # Initialize data loader
    loader = DataLoader()
    
    # Load datasets
    file_paths = ['Cu_Fe_DB.csv', 'database (1).csv']
    datasets = loader.load_datasets(file_paths)
    
    if not datasets:
        print("No datasets loaded successfully!")
        return
    
    # Merge datasets with duplicate detection
    merged_data = loader.merge_datasets(datasets)
    
    # Handle missing compressive strength values
    cleaned_data = loader.handle_missing_compressive_strength(merged_data)
    
    # Extract element fractions
    data_with_elements = loader.extract_element_fractions(cleaned_data)
    
    # Final cleaning and validation
    final_data = loader.clean_and_validate_data(data_with_elements)
    
    # Save processed data
    loader.save_processed_data(final_data)
    
    # Print merge report
    report = loader.generate_merge_report()
    print(f"\nMerge Report:")
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()