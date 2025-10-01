import pandas as pd
import re

# Load original datasets
df1 = pd.read_csv('Cu_Fe_DB.csv')
df2 = pd.read_csv('database (1).csv')

print('=== ORIGINAL DATA ANALYSIS ===')
print(f'Cu_Fe_DB.csv: {len(df1)} rows')
print(f'database (1).csv: {len(df2)} rows')

# Check compressive strength availability
cs1 = df1['Compressive strength (MPa)'].notna().sum()
cs2 = df2['σ (MPa) at 23 C'].notna().sum() if 'σ (MPa) at 23 C' in df2.columns else 0

print(f'Cu_Fe_DB.csv with compressive strength: {cs1}')
print(f'database (1).csv with compressive strength: {cs2}')
print(f'Total potential samples: {cs1 + cs2}')

# Check DOI overlap
def standardize_doi(doi):
    if pd.isna(doi) or doi == '':
        return ''
    doi_str = str(doi).strip().lower()
    doi_str = re.sub(r'^(https?://)?(dx\.)?doi\.org/', '', doi_str)
    doi_str = re.sub(r'^doi:', '', doi_str)
    return doi_str

dois1 = set(standardize_doi(doi) for doi in df1['Source'].dropna() if standardize_doi(doi))
dois2 = set(standardize_doi(doi) for doi in df2['doi'].dropna() if standardize_doi(doi))

print(f'Unique DOIs in Cu_Fe_DB.csv: {len(dois1)}')
print(f'Unique DOIs in database (1).csv: {len(dois2)}')
print(f'Overlapping DOIs: {len(dois1.intersection(dois2))}')

# Check what's happening in the current processed data
df_processed = pd.read_csv('processed_data.csv')
print(f'\nProcessed data: {len(df_processed)} rows')

# Let's see which DOIs are causing the massive reduction
print('\nDOI frequency analysis:')
combined_df = pd.concat([
    df1.assign(dataset='Cu_Fe_DB')[['Source', 'dataset']].rename(columns={'Source': 'doi'}),
    df2.assign(dataset='database_1')[['doi', 'dataset']]
], ignore_index=True)

combined_df['doi_std'] = combined_df['doi'].apply(standardize_doi)
doi_counts = combined_df[combined_df['doi_std'] != '']['doi_std'].value_counts()
print(f'DOIs appearing more than once: {(doi_counts > 1).sum()}')
print(f'Total records with these DOIs: {doi_counts[doi_counts > 1].sum()}')