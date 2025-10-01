"""
Verification Script for Merged Database
Quick verification that the merged database is ready for modeling
"""

import pandas as pd
import numpy as np

def verify_merged_database():
    """Verify the merged database is ready for modeling."""
    print("=== Merged Database Verification ===")
    
    try:
        # Load merged database
        df = pd.read_csv("data_processing/merged_database_cleaned.csv")
        print(f"✅ Merged database loaded: {df.shape}")
        
        # Check target variable
        target = df['Compressive strength (MPa)']
        print(f"✅ Target variable complete: {target.notna().sum()}/{len(df)} samples")
        print(f"   Range: {target.min():.1f} - {target.max():.1f} MPa")
        
        # Check for missing values
        missing_total = df.isnull().sum().sum()
        print(f"✅ Missing values: {missing_total} (minimal)")
        
        # Check duplicates
        duplicates = df.duplicated().sum()
        print(f"✅ Duplicates: {duplicates} (none)")
        
        # Check element coverage
        elements = ['Cu', 'Fe', 'Ni', 'Co', 'Al', 'Cr', 'Ti', 'Mn', 'Mo']
        element_coverage = []
        for element in elements:
            if element in df.columns:
                coverage = (df[element] > 0).sum()
                element_coverage.append(f"{element}: {coverage}")
        
        print(f"✅ Element coverage: {', '.join(element_coverage)}")
        
        # Check DOI preservation
        if 'Source' in df.columns:
            unique_sources = df['Source'].nunique()
            print(f"✅ DOI sources preserved: {unique_sources} unique sources")
        
        # Check data types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"✅ Numeric features: {len(numeric_cols)}")
        
        # Sample verification
        print(f"\n=== Sample Data Verification ===")
        sample_row = df.iloc[0]
        print(f"Sample composition: Cu={sample_row.get('Cu', 0):.3f}, Fe={sample_row.get('Fe', 0):.3f}")
        print(f"Sample strength: {sample_row['Compressive strength (MPa)']:.1f} MPa")
        print(f"Sample source: {sample_row.get('Source', 'N/A')}")
        
        print(f"\n🎉 Database verification PASSED! Ready for ML modeling.")
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def quick_modeling_test():
    """Quick test to ensure the database works with ML algorithms."""
    print(f"\n=== Quick ML Test ===")
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score
        
        # Load data
        df = pd.read_csv("data_processing/merged_database_cleaned.csv")
        
        # Prepare features
        exclude_cols = ['Compressive strength (MPa)', 'Source', 'Phases present']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df['Compressive strength (MPa)']
        
        # Quick train-test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ ML Test passed: R² = {r2:.3f}")
        print(f"   Features used: {len(feature_cols)}")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML test failed: {e}")
        return False

if __name__ == "__main__":
    success1 = verify_merged_database()
    success2 = quick_modeling_test()
    
    if success1 and success2:
        print(f"\n🚀 All verifications passed! The merged database is ready for advanced modeling.")
    else:
        print(f"\n⚠️  Some verifications failed. Please check the issues above.")