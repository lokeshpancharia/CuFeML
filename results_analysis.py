#!/usr/bin/env python3
"""
Results Analysis Script for Cu-Fe HEA Compressive Strength Prediction

This script loads trained models and generates comprehensive analysis including:
- Feature importance rankings from the best performing model
- Predictions vs actual values plots
- Feature relationship analysis
- Correlation analysis and residual plots
- Model performance visualizations
"""

import pandas as pd
import numpy as np
import pickle
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Statistical analysis
from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ResultsAnalyzer:
    """
    Comprehensive results analysis class for Cu-Fe HEA model evaluation.
    """
    
    def __init__(self, features_path='features_data.csv', results_path='results/model_comparison_results.json'):
        """
        Initialize the results analyzer.
        
        Args:
            features_path (str): Path to the features data CSV file
            results_path (str): Path to the model comparison results JSON file
        """
        self.features_path = features_path
        self.results_path = results_path
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.X_test = None
        self.y_test = None
        
        # Create results directory for analysis outputs
        Path('results/analysis').mkdir(parents=True, exist_ok=True)
        
    def load_data_and_models(self):
        """Load the features data, model results, and trained models."""
        print("Loading data and models...")
        
        # Load features data
        self.df = pd.read_csv(self.features_path)
        print(f"Loaded dataset with shape: {self.df.shape}")
        
        # Prepare features and target (same logic as in training)
        target_col = 'compressive_strength'
        exclude_cols = [target_col, 'Test Type']
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        self.X = self.df[feature_cols].copy()
        self.y = self.df[target_col].copy()
        self.feature_names = feature_cols
        
        # Handle missing values
        if self.X.isnull().sum().sum() > 0:
            self.X = self.X.fillna(self.X.median())
        
        # Load model results
        with open(self.results_path, 'r') as f:
            self.results = json.load(f)
        
        # Get best model info
        self.best_model_name = self.results['metadata']['best_model']
        print(f"Best model: {self.best_model_name}")
        
        # Load the best model
        best_model_path = f"models/{self.best_model_name.lower().replace(' ', '_')}_model.pkl"
        with open(best_model_path, 'rb') as f:
            self.best_model = pickle.load(f)
        
        # Load all models for comparison
        model_names = [name for name in self.results.keys() if name != 'metadata']
        for model_name in model_names:
            model_path = f"models/{model_name.lower().replace(' ', '_')}_model.pkl"
            try:
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
            except FileNotFoundError:
                print(f"Warning: Model file not found for {model_name}")
        
        # Recreate test split (same random state as training)
        from sklearn.model_selection import train_test_split
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        print(f"Test set size: {len(self.X_test)}")
        
    def extract_feature_importance(self):
        """Extract and analyze feature importance from the best model."""
        print(f"\nExtracting feature importance from {self.best_model_name}...")
        
        # Get feature importance based on model type
        if hasattr(self.best_model, 'feature_importances_'):
            # Tree-based models (Random Forest, XGBoost)
            if hasattr(self.best_model, 'named_steps'):
                # Pipeline model
                regressor = self.best_model.named_steps['regressor']
                importance_scores = regressor.feature_importances_
            else:
                importance_scores = self.best_model.feature_importances_
        else:
            # For models without feature_importances_ (like SVR), use permutation importance
            from sklearn.inspection import permutation_importance
            perm_importance = permutation_importance(
                self.best_model, self.X_test, self.y_test, 
                n_repeats=10, random_state=42
            )
            importance_scores = perm_importance.importances_mean
        
        # Create feature importance DataFrame
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        print("Top 10 most important features:")
        print(self.feature_importance.head(10))
        
        # Save feature importance
        self.feature_importance.to_csv('results/analysis/feature_importance.csv', index=False)
        
        return self.feature_importance
    
    def create_predictions_vs_actual_plot(self):
        """Create predictions vs actual values plot for all models."""
        print("\nCreating predictions vs actual plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        model_names = list(self.models.keys())
        
        for i, model_name in enumerate(model_names):
            if i >= 4:  # Only plot first 4 models
                break
                
            model = self.models[model_name]
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # Create scatter plot
            axes[i].scatter(self.y_test, y_pred, alpha=0.6, s=50)
            
            # Add perfect prediction line
            min_val = min(self.y_test.min(), y_pred.min())
            max_val = max(self.y_test.max(), y_pred.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.8)
            
            # Formatting
            axes[i].set_xlabel('Actual Compressive Strength (MPa)')
            axes[i].set_ylabel('Predicted Compressive Strength (MPa)')
            axes[i].set_title(f'{model_name}\nRMSE: {rmse:.1f}, MAE: {mae:.1f}, R²: {r2:.3f}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/analysis/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_feature_importance_plot(self):
        """Create feature importance visualization."""
        print("\nCreating feature importance plot...")
        
        # Plot top 20 features
        top_features = self.feature_importance.head(20)
        
        plt.figure(figsize=(12, 10))
        bars = plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 20 Feature Importance - {self.best_model_name}')
        plt.gca().invert_yaxis()
        
        # Color bars by importance level
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        plt.savefig('results/analysis/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_residual_plots(self):
        """Create residual analysis plots for the best model."""
        print("\nCreating residual analysis plots...")
        
        y_pred = self.best_model.predict(self.X_test)
        residuals = self.y_test - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted Values')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals vs Actual
        axes[0, 1].scatter(self.y_test, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Actual Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Actual Values')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/analysis/residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def analyze_feature_correlations(self):
        """Analyze correlations between features and target variable."""
        print("\nAnalyzing feature correlations...")
        
        # Calculate correlations with target
        correlations = []
        for feature in self.feature_names:
            corr, p_value = pearsonr(self.X[feature], self.y)
            correlations.append({
                'feature': feature,
                'correlation': corr,
                'abs_correlation': abs(corr),
                'p_value': p_value
            })
        
        correlation_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
        
        # Save correlations
        correlation_df.to_csv('results/analysis/feature_correlations.csv', index=False)
        
        # Plot top correlations
        top_corr = correlation_df.head(20)
        
        plt.figure(figsize=(12, 10))
        colors = ['red' if x < 0 else 'blue' for x in top_corr['correlation']]
        bars = plt.barh(range(len(top_corr)), top_corr['correlation'], color=colors, alpha=0.7)
        plt.yticks(range(len(top_corr)), top_corr['feature'])
        plt.xlabel('Correlation with Compressive Strength')
        plt.title('Top 20 Feature Correlations with Target Variable')
        plt.gca().invert_yaxis()
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/analysis/feature_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Top 10 correlated features:")
        print(correlation_df.head(10)[['feature', 'correlation', 'p_value']])
        
        return correlation_df
    
    def create_model_comparison_plot(self):
        """Create model performance comparison visualization."""
        print("\nCreating model comparison plot...")
        
        # Extract metrics for all models
        model_metrics = []
        for model_name in self.models.keys():
            if model_name in self.results:
                metrics = self.results[model_name]
                model_metrics.append({
                    'Model': model_name,
                    'CV RMSE': metrics['cv_rmse_mean'],
                    'Test RMSE': metrics['test_rmse'],
                    'Test MAE': metrics['test_mae'],
                    'Test R²': metrics['test_r2']
                })
        
        metrics_df = pd.DataFrame(model_metrics)
        
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # CV RMSE
        axes[0, 0].bar(metrics_df['Model'], metrics_df['CV RMSE'], alpha=0.7)
        axes[0, 0].set_title('Cross-Validation RMSE')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Test RMSE
        axes[0, 1].bar(metrics_df['Model'], metrics_df['Test RMSE'], alpha=0.7, color='orange')
        axes[0, 1].set_title('Test RMSE')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Test MAE
        axes[1, 0].bar(metrics_df['Model'], metrics_df['Test MAE'], alpha=0.7, color='green')
        axes[1, 0].set_title('Test MAE')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Test R²
        axes[1, 1].bar(metrics_df['Model'], metrics_df['Test R²'], alpha=0.7, color='red')
        axes[1, 1].set_title('Test R²')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('results/analysis/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metrics comparison
        metrics_df.to_csv('results/analysis/model_metrics_comparison.csv', index=False)
        
        return metrics_df
    
    def analyze_prediction_errors(self):
        """Analyze prediction errors and identify patterns."""
        print("\nAnalyzing prediction errors...")
        
        y_pred = self.best_model.predict(self.X_test)
        errors = np.abs(self.y_test - y_pred)
        relative_errors = errors / self.y_test * 100
        
        # Create error analysis DataFrame
        error_analysis = pd.DataFrame({
            'actual': self.y_test,
            'predicted': y_pred,
            'absolute_error': errors,
            'relative_error': relative_errors
        })
        
        # Add feature values for error analysis
        for feature in self.feature_names[:10]:  # Top 10 features only
            error_analysis[feature] = self.X_test[feature].values
        
        # Sort by error magnitude
        error_analysis = error_analysis.sort_values('absolute_error', ascending=False)
        
        # Save error analysis
        error_analysis.to_csv('results/analysis/prediction_errors.csv', index=False)
        
        # Plot error distribution
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Absolute errors
        axes[0].hist(errors, bins=20, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Absolute Error (MPa)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Absolute Errors')
        axes[0].grid(True, alpha=0.3)
        
        # Relative errors
        axes[1].hist(relative_errors, bins=20, alpha=0.7, edgecolor='black', color='orange')
        axes[1].set_xlabel('Relative Error (%)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Relative Errors')
        axes[1].grid(True, alpha=0.3)
        
        # Error vs actual values
        axes[2].scatter(self.y_test, errors, alpha=0.6)
        axes[2].set_xlabel('Actual Compressive Strength (MPa)')
        axes[2].set_ylabel('Absolute Error (MPa)')
        axes[2].set_title('Error vs Actual Values')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/analysis/error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Error Statistics:")
        print(f"Mean Absolute Error: {errors.mean():.2f} MPa")
        print(f"Median Absolute Error: {np.median(errors):.2f} MPa")
        print(f"Mean Relative Error: {relative_errors.mean():.2f}%")
        print(f"Median Relative Error: {np.median(relative_errors):.2f}%")
        
        return error_analysis
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report."""
        print("\nGenerating comprehensive analysis report...")
        
        report = []
        report.append("="*80)
        report.append("Cu-Fe HEA COMPRESSIVE STRENGTH PREDICTION - RESULTS ANALYSIS")
        report.append("="*80)
        report.append("")
        
        # Dataset summary
        report.append("DATASET SUMMARY")
        report.append("-" * 40)
        report.append(f"Total samples: {len(self.df)}")
        report.append(f"Features: {len(self.feature_names)}")
        report.append(f"Test samples: {len(self.X_test)}")
        report.append(f"Target range: {self.y.min():.1f} - {self.y.max():.1f} MPa")
        report.append(f"Target mean: {self.y.mean():.1f} MPa")
        report.append(f"Target std: {self.y.std():.1f} MPa")
        report.append("")
        
        # Best model performance
        report.append("BEST MODEL PERFORMANCE")
        report.append("-" * 40)
        report.append(f"Best model: {self.best_model_name}")
        best_results = self.results[self.best_model_name]
        report.append(f"Cross-validation RMSE: {best_results['cv_rmse_mean']:.2f} ± {best_results['cv_rmse_std']:.2f}")
        report.append(f"Test RMSE: {best_results['test_rmse']:.2f}")
        report.append(f"Test MAE: {best_results['test_mae']:.2f}")
        report.append(f"Test R²: {best_results['test_r2']:.3f}")
        report.append("")
        
        # Feature importance summary
        report.append("TOP 10 MOST IMPORTANT FEATURES")
        report.append("-" * 40)
        for i, row in self.feature_importance.head(10).iterrows():
            report.append(f"{row['feature']}: {row['importance']:.4f}")
        report.append("")
        
        # Model comparison
        report.append("MODEL COMPARISON")
        report.append("-" * 40)
        for model_name in self.models.keys():
            if model_name in self.results:
                results = self.results[model_name]
                report.append(f"{model_name}:")
                report.append(f"  Test RMSE: {results['test_rmse']:.2f}")
                report.append(f"  Test R²: {results['test_r2']:.3f}")
        report.append("")
        
        # Key insights
        report.append("KEY INSIGHTS")
        report.append("-" * 40)
        
        # Analyze top features for materials science insights
        top_features = self.feature_importance.head(5)['feature'].tolist()
        composition_features = [f for f in top_features if any(element in f for element in ['Cu', 'Fe', 'Al', 'Co', 'Cr', 'Ni'])]
        phase_features = [f for f in top_features if 'phase' in f.lower() or any(phase in f for phase in ['FCC', 'BCC', 'HCP'])]
        derived_features = [f for f in top_features if any(term in f for term in ['ratio', 'interaction', 'entropy', 'radius'])]
        
        if composition_features:
            report.append(f"• Composition features dominate: {', '.join(composition_features[:3])}")
        if phase_features:
            report.append(f"• Phase structure is important: {', '.join(phase_features[:2])}")
        if derived_features:
            report.append(f"• Derived features provide value: {', '.join(derived_features[:2])}")
        
        # Performance insights
        rmse_percentage = (best_results['test_rmse'] / self.y.mean()) * 100
        report.append(f"• Model achieves {rmse_percentage:.1f}% relative error on average")
        
        if best_results['test_r2'] > 0.7:
            report.append("• Excellent model performance (R² > 0.7)")
        elif best_results['test_r2'] > 0.5:
            report.append("• Good model performance (R² > 0.5)")
        else:
            report.append("• Moderate model performance - consider feature engineering")
        
        report.append("")
        report.append("FILES GENERATED")
        report.append("-" * 40)
        report.append("• results/analysis/feature_importance.csv")
        report.append("• results/analysis/feature_correlations.csv")
        report.append("• results/analysis/model_metrics_comparison.csv")
        report.append("• results/analysis/prediction_errors.csv")
        report.append("• results/analysis/predictions_vs_actual.png")
        report.append("• results/analysis/feature_importance.png")
        report.append("• results/analysis/feature_correlations.png")
        report.append("• results/analysis/residual_analysis.png")
        report.append("• results/analysis/model_comparison.png")
        report.append("• results/analysis/error_analysis.png")
        report.append("")
        
        # Save report
        report_text = "\n".join(report)
        with open('results/analysis/comprehensive_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        
    def run_complete_analysis(self):
        """Run the complete results analysis pipeline."""
        print("Starting Cu-Fe HEA Compressive Strength Results Analysis")
        print("="*80)
        
        try:
            # Load data and models
            self.load_data_and_models()
            
            # Extract feature importance
            self.extract_feature_importance()
            
            # Create visualizations
            self.create_predictions_vs_actual_plot()
            self.create_feature_importance_plot()
            self.create_residual_plots()
            
            # Analyze correlations
            self.analyze_feature_correlations()
            
            # Model comparison
            self.create_model_comparison_plot()
            
            # Error analysis
            self.analyze_prediction_errors()
            
            # Generate comprehensive report
            self.generate_comprehensive_report()
            
            print("\nResults analysis completed successfully!")
            print("All visualizations and analysis files saved to results/analysis/")
            
        except Exception as e:
            print(f"Error during results analysis: {str(e)}")
            raise


def main():
    """Main function to run the results analysis."""
    analyzer = ResultsAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()