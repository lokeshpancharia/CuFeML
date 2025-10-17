"""
Diagnostic Analysis for Compressive Strength Prediction
Investigate why R² scores are lower than expected and identify issues
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

class DiagnosticAnalyzer:
    def __init__(self, data_path="data_processing/merged_database_cleaned.csv"):
        """Initialize diagnostic analyzer."""
        self.data_path = data_path
        self.df = None
        
    def load_and_analyze_data(self):
        """Load data and perform basic analysis."""
        print("=== Data Loading and Basic Analysis ===")
        
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        
        # Analyze target variable
        target = self.df['Compressive strength (MPa)']
        print(f"\nTarget Variable Analysis:")
        print(f"Range: {target.min():.1f} - {target.max():.1f} MPa")
        print(f"Mean: {target.mean():.1f} MPa")
        print(f"Median: {target.median():.1f} MPa")
        print(f"Std: {target.std():.1f} MPa")
        print(f"Coefficient of Variation: {target.std()/target.mean():.3f}")
        
        # Check for outliers
        Q1 = target.quantile(0.25)
        Q3 = target.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((target < lower_bound) | (target > upper_bound)).sum()
        print(f"Outliers (IQR method): {outliers} ({outliers/len(target)*100:.1f}%)")
        
        # Analyze distribution
        from scipy import stats
        shapiro_stat, shapiro_p = stats.shapiro(target.sample(min(5000, len(target))))
        print(f"Shapiro-Wilk normality test: stat={shapiro_stat:.4f}, p={shapiro_p:.6f}")
        
        return self.df
    
    def analyze_feature_quality(self):
        """Analyze feature quality and relationships."""
        print("\n=== Feature Quality Analysis ===")
        
        # Get numeric features
        exclude_cols = ['Compressive strength (MPa)', 'Source', 'Phases present', 'Unique ID', 'Composition']
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        X = self.df[feature_cols].fillna(self.df[feature_cols].median())
        y = self.df['Compressive strength (MPa)']
        
        print(f"Number of features: {len(feature_cols)}")
        
        # Analyze feature correlations with target
        correlations = []
        for col in feature_cols:
            corr = abs(X[col].corr(y))
            correlations.append((col, corr))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 10 features by correlation with target:")
        for i, (feature, corr) in enumerate(correlations[:10], 1):
            print(f"{i:2d}. {feature}: {corr:.4f}")
        
        # Check for multicollinearity
        print(f"\nMulticollinearity Analysis:")
        corr_matrix = X[feature_cols].corr().abs()
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.9:
                    high_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_matrix.iloc[i, j]
                    ))
        
        print(f"Highly correlated feature pairs (>0.9): {len(high_corr_pairs)}")
        for feat1, feat2, corr in high_corr_pairs[:5]:
            print(f"  {feat1} - {feat2}: {corr:.4f}")
        
        return X, y, correlations
    
    def analyze_data_splits(self, X, y):
        """Analyze how data splits affect performance."""
        print("\n=== Data Split Analysis ===")
        
        # Test different split strategies
        split_results = {}
        
        for test_size in [0.1, 0.2, 0.3]:
            for random_state in [42, 123, 456]:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                
                # Simple RF model
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X_train, y_train)
                
                train_score = rf.score(X_train, y_train)
                test_score = rf.score(X_test, y_test)
                
                key = f"test_size_{test_size}_seed_{random_state}"
                split_results[key] = {
                    'train_r2': train_score,
                    'test_r2': test_score,
                    'overfitting': train_score - test_score
                }
        
        # Analyze results
        avg_train_r2 = np.mean([r['train_r2'] for r in split_results.values()])
        avg_test_r2 = np.mean([r['test_r2'] for r in split_results.values()])
        avg_overfitting = np.mean([r['overfitting'] for r in split_results.values()])
        
        print(f"Average training R²: {avg_train_r2:.4f}")
        print(f"Average test R²: {avg_test_r2:.4f}")
        print(f"Average overfitting: {avg_overfitting:.4f}")
        
        return split_results
    
    def analyze_sample_size_effect(self, X, y):
        """Analyze how sample size affects performance."""
        print("\n=== Sample Size Effect Analysis ===")
        
        sample_sizes = [50, 100, 150, 200, len(X)]
        results = []
        
        for n_samples in sample_sizes:
            if n_samples > len(X):
                continue
                
            # Sample data
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X.iloc[indices]
            y_sample = y.iloc[indices]
            
            # Cross-validation
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            cv_scores = cross_val_score(rf, X_sample, y_sample, cv=5, scoring='r2')
            
            results.append({
                'n_samples': n_samples,
                'mean_cv_r2': cv_scores.mean(),
                'std_cv_r2': cv_scores.std()
            })
            
            print(f"n={n_samples}: CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return results
    
    def analyze_feature_importance_stability(self, X, y):
        """Analyze stability of feature importance across different splits."""
        print("\n=== Feature Importance Stability Analysis ===")
        
        importance_results = {}
        
        for i in range(5):  # 5 different random splits
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=i
            )
            
            rf = RandomForestRegressor(n_estimators=200, random_state=42)
            rf.fit(X_train, y_train)
            
            # Get feature importance
            for j, feature in enumerate(X.columns):
                if feature not in importance_results:
                    importance_results[feature] = []
                importance_results[feature].append(rf.feature_importances_[j])
        
        # Calculate stability metrics
        stability_metrics = {}
        for feature, importances in importance_results.items():
            stability_metrics[feature] = {
                'mean': np.mean(importances),
                'std': np.std(importances),
                'cv': np.std(importances) / np.mean(importances) if np.mean(importances) > 0 else np.inf
            }
        
        # Sort by mean importance
        sorted_features = sorted(stability_metrics.items(), key=lambda x: x[1]['mean'], reverse=True)
        
        print("Top 10 most important features (with stability):")
        for i, (feature, metrics) in enumerate(sorted_features[:10], 1):
            print(f"{i:2d}. {feature}: {metrics['mean']:.4f} ± {metrics['std']:.4f} (CV: {metrics['cv']:.2f})")
        
        return stability_metrics
    
    def analyze_target_distribution_by_source(self):
        """Analyze target distribution by data source."""
        print("\n=== Target Distribution by Source Analysis ===")
        
        if 'Source' not in self.df.columns:
            print("No Source column found")
            return
        
        target = self.df['Compressive strength (MPa)']
        
        # Group by source
        source_stats = self.df.groupby('Source')[target.name].agg(['count', 'mean', 'std', 'min', 'max'])
        source_stats = source_stats.sort_values('count', ascending=False)
        
        print("Target statistics by source (top 10):")
        print(source_stats.head(10).round(2))
        
        # Check for significant differences between sources
        from scipy.stats import f_oneway
        
        # Get sources with at least 5 samples
        large_sources = source_stats[source_stats['count'] >= 5].index
        if len(large_sources) >= 2:
            source_groups = [self.df[self.df['Source'] == source][target.name].values 
                           for source in large_sources]
            
            f_stat, p_value = f_oneway(*source_groups)
            print(f"\nANOVA test for differences between sources:")
            print(f"F-statistic: {f_stat:.4f}, p-value: {p_value:.6f}")
            
            if p_value < 0.05:
                print("Significant differences between sources detected!")
            else:
                print("No significant differences between sources.")
        
        return source_stats
    
    def identify_problematic_samples(self, X, y):
        """Identify potentially problematic samples."""
        print("\n=== Problematic Sample Analysis ===")
        
        # Train a model and identify samples with high residuals
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        
        # Predict on all data
        y_pred_all = rf.predict(X)
        residuals = np.abs(y - y_pred_all)
        
        # Identify high-residual samples
        residual_threshold = np.percentile(residuals, 95)  # Top 5% residuals
        problematic_indices = np.where(residuals > residual_threshold)[0]
        
        print(f"Identified {len(problematic_indices)} potentially problematic samples (top 5% residuals)")
        
        # Analyze these samples
        problematic_samples = self.df.iloc[problematic_indices]
        
        print("Problematic samples analysis:")
        print(f"Target range: {problematic_samples['Compressive strength (MPa)'].min():.1f} - {problematic_samples['Compressive strength (MPa)'].max():.1f}")
        
        if 'Source' in problematic_samples.columns:
            source_counts = problematic_samples['Source'].value_counts()
            print("Sources of problematic samples:")
            print(source_counts.head())
        
        return problematic_indices, residuals
    
    def generate_diagnostic_report(self):
        """Generate comprehensive diagnostic report."""
        print("\n=== Generating Diagnostic Report ===")
        
        report = []
        report.append("DIAGNOSTIC ANALYSIS REPORT - Compressive Strength Prediction")
        report.append("=" * 70)
        
        # Load and analyze data
        df = self.load_and_analyze_data()
        X, y, correlations = self.analyze_feature_quality()
        
        # Key findings
        report.append("\nKEY FINDINGS:")
        report.append("-" * 20)
        
        # Target variable analysis
        cv = y.std() / y.mean()
        if cv > 0.5:
            report.append(f"- High target variability (CV = {cv:.3f}) makes prediction challenging")
        
        # Feature correlation analysis
        max_corr = max([corr for _, corr in correlations])
        if max_corr < 0.6:
            report.append(f"- Weak feature-target correlations (max = {max_corr:.3f})")
        
        # Sample size analysis
        n_samples = len(df)
        n_features = len(X.columns)
        ratio = n_features / n_samples
        if ratio > 0.3:
            report.append(f"- High feature-to-sample ratio ({ratio:.2f}) may cause overfitting")
        
        # Data split analysis
        split_results = self.analyze_data_splits(X, y)
        avg_overfitting = np.mean([r['overfitting'] for r in split_results.values()])
        if avg_overfitting > 0.3:
            report.append(f"- High overfitting detected (avg = {avg_overfitting:.3f})")
        
        # Recommendations
        report.append("\nRECOMMendations:")
        report.append("-" * 20)
        
        if cv > 0.5:
            report.append("- Consider target transformation or stratified sampling")
        
        if max_corr < 0.6:
            report.append("- Need better feature engineering or domain expertise")
        
        if ratio > 0.3:
            report.append("- Apply aggressive feature selection or regularization")
        
        if avg_overfitting > 0.3:
            report.append("- Use stronger regularization and cross-validation")
        
        report.append("- Collect more high-quality training data")
        report.append("- Investigate data quality and outliers")
        report.append("- Consider ensemble methods with different algorithms")
        
        # Save report
        report_text = '\n'.join(report)
        with open('data_processing/diagnostic_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        return report_text

def main():
    """Main diagnostic analysis."""
    print("DIAGNOSTIC ANALYSIS FOR COMPRESSIVE STRENGTH PREDICTION")
    print("Investigating why R² scores are lower than expected")
    print("=" * 65)
    
    analyzer = DiagnosticAnalyzer()
    
    # Load and analyze data
    df = analyzer.load_and_analyze_data()
    X, y, correlations = analyzer.analyze_feature_quality()
    
    # Various analyses
    split_results = analyzer.analyze_data_splits(X, y)
    sample_size_results = analyzer.analyze_sample_size_effect(X, y)
    stability_metrics = analyzer.analyze_feature_importance_stability(X, y)
    source_stats = analyzer.analyze_target_distribution_by_source()
    problematic_indices, residuals = analyzer.identify_problematic_samples(X, y)
    
    # Generate comprehensive report
    analyzer.generate_diagnostic_report()
    
    print("\nDIAGNOSTIC ANALYSIS COMPLETE!")
    print("Check 'data_processing/diagnostic_report.txt' for detailed findings")

if __name__ == "__main__":
    main()