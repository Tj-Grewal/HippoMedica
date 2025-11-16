#!/usr/bin/env python3
"""
Step 2: Exploratory Data Analysis
Comprehensive analysis of the diabetes dataset with visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ExploratoryAnalysis:
    """Performs comprehensive EDA on disease datasets."""
    
    def __init__(self):
        self.base_data_dir = Path(__file__).parent / "../datasets/raw"
        self.output_base_dir = Path(__file__).parent / "outputs" / "visualizations"
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Define dataset configurations
        self.dataset_configs = {
            'diabetes': {
                'filename': 'diabetes.csv',
                'target_column': 'Outcome',
                'target_labels': {0: 'No Diabetes', 1: 'Diabetes'},
                'zero_check_cols': ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            },
            'heart_disease': {
                'filename': 'heart.csv',
                'target_column': 'target',
                'target_labels': {0: 'No Heart Disease', 1: 'Heart Disease'},
                'zero_check_cols': ['trestbps', 'chol', 'thalach']
            },
            'stroke': {
                'filename': 'healthcare-dataset-stroke-data.csv',
                'target_column': 'stroke',
                'target_labels': {0: 'No Stroke', 1: 'Stroke'},
                'zero_check_cols': ['bmi', 'avg_glucose_level']
            }
        }
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        self.dataset = None
        self.config = None
        self.disease_key = None
        self.eda_summary = {}
    
    def load_data(self, disease_key):
        """Load a disease dataset.
        
        Args:
            disease_key: One of 'diabetes', 'heart_disease', 'stroke'
        """
        if disease_key not in self.dataset_configs:
            print(f"[X] Unknown disease: {disease_key}")
            return False
        
        self.disease_key = disease_key
        self.config = self.dataset_configs[disease_key]
        data_path = self.base_data_dir / self.config['filename']
        self.output_dir = self.output_base_dir / disease_key
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[>] Loading {disease_key} data from: {data_path}")
        
        try:
            self.dataset = pd.read_csv(data_path)
            print(f"[OK] Loaded dataset: {self.dataset.shape[0]} rows, {self.dataset.shape[1]} columns")
            return True
            
        except Exception as e:
            print(f"[X] Error loading data: {e}")
            return False
    
    def basic_statistics(self):
        """Calculate and display basic statistics."""
        print("\n" + "="*80)
        print("BASIC STATISTICS")
        print("="*80)
        
        print(f"\n[#] Dataset Shape: {self.dataset.shape}")
        print(f"   Rows: {self.dataset.shape[0]:,}")
        print(f"   Columns: {self.dataset.shape[1]}")
        
        target_col = self.config['target_column']
        target_labels = self.config['target_labels']
        
        print(f"\n[*] Target Distribution ({target_col}):")
        outcome_counts = self.dataset[target_col].value_counts()
        for key in sorted(outcome_counts.index):
            label = target_labels.get(key, f"Class {key}")
            count = outcome_counts[key]
            pct = count / len(self.dataset) * 100
            print(f"   {label}: {count} ({pct:.1f}%)")
        
        # Show statistics for numeric columns only
        print(f"\n[^] Numeric Feature Statistics:")
        numeric_data = self.dataset.select_dtypes(include=[np.number])
        print(numeric_data.describe().round(2))
        
        # Show info about categorical columns
        categorical_cols = self.dataset.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            print(f"\n[^] Categorical Features ({len(categorical_cols)} columns):")
            for col in categorical_cols:
                unique_count = self.dataset[col].nunique()
                top_value = self.dataset[col].value_counts().index[0] if len(self.dataset[col]) > 0 else "N/A"
                print(f"   {col}: {unique_count} unique values (most common: {top_value})")
        
        # Store summary
        self.eda_summary['shape'] = self.dataset.shape
        self.eda_summary['target_distribution'] = outcome_counts.to_dict()
        self.eda_summary['statistics'] = numeric_data.describe().to_dict()  # Only store numeric stats
        if categorical_cols:
            self.eda_summary['categorical_info'] = {
                col: {
                    'unique_count': int(self.dataset[col].nunique()),
                    'top_value': str(self.dataset[col].value_counts().index[0]) if len(self.dataset[col]) > 0 else "N/A"
                }
                for col in categorical_cols
            }
    
    def missing_values_analysis(self):
        """Analyze missing values and zeros in medical features."""
        print("\n" + "="*80)
        print("MISSING VALUES & ZERO ANALYSIS")
        print("="*80)
        
        # Separate numeric and categorical columns
        numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.dataset.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Check for NaN values in all columns
        missing = self.dataset.isnull().sum()
        if missing.sum() > 0:
            print(f"\n[!] Missing values (NaN) found:")
            for col, count in missing[missing > 0].items():
                col_type = "numeric" if col in numeric_cols else "categorical"
                print(f"   {col} ({col_type}): {count} ({count/len(self.dataset)*100:.1f}%)")
        else:
            print(f"\n[OK] No NaN values found")
        
        # Show categorical columns info (not missing, just non-numeric)
        if categorical_cols:
            print(f"\n[i] Categorical columns (text data, not missing):")
            for col in categorical_cols:
                if col not in missing or missing[col] == 0:
                    unique_vals = self.dataset[col].nunique()
                    print(f"   {col}: {unique_vals} unique values")
        
        # Check for zeros (likely missing values in medical data)
        zero_check_cols = self.config['zero_check_cols']
        print(f"\n[?] Zero values in medical features (likely missing data):")
        
        zero_summary = {}
        for col in zero_check_cols:
            if col in self.dataset.columns and col in numeric_cols:
                zero_count = (self.dataset[col] == 0).sum()
                zero_pct = (zero_count / len(self.dataset)) * 100
                print(f"   {col}: {zero_count} zeros ({zero_pct:.1f}%)")
                zero_summary[col] = {'count': int(zero_count), 'percent': float(zero_pct)}
        
        self.eda_summary['zero_values'] = zero_summary
        self.eda_summary['categorical_columns'] = categorical_cols
    
    def correlation_analysis(self):
        """Analyze feature correlations."""
        print("\n" + "="*80)
        print("CORRELATION ANALYSIS")
        print("="*80)
        
        # Only use numeric columns
        numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns
        numeric_data = self.dataset[numeric_cols]
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        target_col = self.config['target_column']
        
        # Find features most correlated with outcome
        if target_col in corr_matrix.columns:
            outcome_corr = corr_matrix[target_col].sort_values(ascending=False)
            print(f"\n[#] Features most correlated with {target_col.title()}:")
            for feature, corr_val in outcome_corr[1:].items():  # Skip target itself
                print(f"   {feature}: {corr_val:.3f}")
            
            self.eda_summary['outcome_correlations'] = outcome_corr.to_dict()
        
        # Save correlation heatmap
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={'shrink': 0.8})
        plt.title(f'{self.disease_key.title().replace("_", " ")} - Feature Correlation Heatmap', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / '01_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n[OK] Saved: 01_correlation_heatmap.png")
    
    def distribution_analysis(self):
        """Analyze feature distributions."""
        print("\n" + "="*80)
        print("DISTRIBUTION ANALYSIS")
        print("="*80)
        
        # Get numeric features excluding target
        target_col = self.config['target_column']
        numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns
        features = [col for col in numeric_cols if col != target_col]
        
        if len(features) == 0:
            print("[!] No numeric features found for distribution analysis")
            return
        
        # Create distribution plots
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.ravel()
        
        for idx, feature in enumerate(features):
            ax = axes[idx]
            
            # Histogram with KDE
            self.dataset[feature].dropna().hist(bins=30, ax=ax, alpha=0.7, edgecolor='black')
            ax.set_title(f'{feature} Distribution', fontweight='bold')
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            mean_val = self.dataset[feature].mean()
            median_val = self.dataset[feature].median()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
            ax.legend()
        
        # Hide unused subplots
        for idx in range(len(features), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '02_feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved: 02_feature_distributions.png")
    
    def outcome_comparison(self):
        """Compare features between outcome groups."""
        print("\n" + "="*80)
        print("OUTCOME COMPARISON")
        print("="*80)
        
        target_col = self.config['target_column']
        target_labels = self.config['target_labels']
        
        # Get numeric features excluding target
        numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns
        features = [col for col in numeric_cols if col != target_col]
        
        if len(features) == 0:
            print("[!] No numeric features found for outcome comparison")
            return
        
        # Create box plots comparing outcomes
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.ravel()
        
        for idx, feature in enumerate(features):
            ax = axes[idx]
            
            # Box plot by outcome
            unique_outcomes = sorted(self.dataset[target_col].dropna().unique())
            data_to_plot = []
            labels = []
            
            for outcome in unique_outcomes:
                data = self.dataset[self.dataset[target_col] == outcome][feature].dropna()
                data_to_plot.append(data)
                labels.append(target_labels.get(outcome, f"Class {outcome}"))
            
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # Color the boxes
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_title(f'{feature} by Outcome', fontweight='bold')
            ax.set_ylabel(feature)
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=15)
        
        # Hide unused subplots
        for idx in range(len(features), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '03_outcome_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved: 03_outcome_comparison.png")
        
        # Print mean comparison
        print(f"\n[#] Feature Means by Outcome:")
        for feature in features[:10]:  # Limit to first 10 features for readability
            means = {}
            for outcome in sorted(self.dataset[target_col].dropna().unique()):
                means[outcome] = self.dataset[self.dataset[target_col] == outcome][feature].mean()
            
            output = f"   {feature:25s}"
            for outcome in sorted(means.keys()):
                label = target_labels.get(outcome, f"Class {outcome}")
                output += f" | {label}: {means[outcome]:7.2f}"
            print(output)
    
    def pairwise_relationships(self):
        """Analyze pairwise relationships between key features."""
        print("\n" + "="*80)
        print("PAIRWISE RELATIONSHIPS")
        print("="*80)
        
        target_col = self.config['target_column']
        
        # Get numeric columns
        numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns
        features = [col for col in numeric_cols if col != target_col]
        
        if len(features) < 3:
            print("[!] Not enough numeric features for pairwise analysis")
            return
        
        # Select top 4 features (by correlation with target if available)
        if target_col in self.dataset.columns and 'outcome_correlations' in self.eda_summary:
            correlations = self.eda_summary['outcome_correlations']
            sorted_features = sorted(
                [(f, abs(correlations.get(f, 0))) for f in features if f in correlations],
                key=lambda x: x[1],
                reverse=True
            )
            key_features = [f[0] for f in sorted_features[:4]]
        else:
            key_features = features[:4]
        
        # Create scatter matrix
        n_features = len(key_features)
        fig, axes = plt.subplots(n_features, n_features, figsize=(16, 16))
        
        if n_features == 1:
            axes = np.array([[axes]])
        elif n_features == 2:
            axes = axes.reshape(2, 2)
        
        for i, feat1 in enumerate(key_features):
            for j, feat2 in enumerate(key_features):
                ax = axes[i, j]
                
                if i == j:
                    # Diagonal: distribution by outcome
                    unique_outcomes = sorted(self.dataset[target_col].dropna().unique())
                    for outcome in unique_outcomes:
                        data = self.dataset[self.dataset[target_col] == outcome][feat1].dropna()
                        label = self.config['target_labels'].get(outcome, f'Outcome {outcome}')
                        ax.hist(data, bins=20, alpha=0.6, label=label)
                    ax.set_ylabel('Frequency')
                    if i == 0:
                        ax.legend(fontsize=8)
                else:
                    # Off-diagonal: scatter plot
                    unique_outcomes = sorted(self.dataset[target_col].dropna().unique())
                    colors = ['blue', 'red', 'green', 'orange']
                    for idx, outcome in enumerate(unique_outcomes):
                        data = self.dataset[self.dataset[target_col] == outcome]
                        label = self.config['target_labels'].get(outcome, f'Outcome {outcome}')
                        ax.scatter(data[feat2], data[feat1], 
                                 alpha=0.4, s=10, c=colors[idx % len(colors)],
                                 label=label)
                
                if i == n_features - 1:
                    ax.set_xlabel(feat2, fontsize=9)
                else:
                    ax.set_xticklabels([])
                    
                if j == 0:
                    ax.set_ylabel(feat1, fontsize=9)
                else:
                    ax.set_yticklabels([])
                
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.disease_key.title().replace("_", " ")} - Pairwise Relationships', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / '04_pairwise_relationships.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved: 04_pairwise_relationships.png")
    
    def save_summary_report(self):
        """Save EDA summary report."""
        print("\n" + "="*80)
        print("SAVING SUMMARY REPORT")
        print("="*80)
        
        report_path = self.output_dir / f"eda_summary_{self.disease_key}.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"EXPLORATORY DATA ANALYSIS SUMMARY - {self.disease_key.upper()}\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Dataset Shape: {self.eda_summary['shape']}\n")
            f.write(f"\nTarget Distribution:\n")
            for key, val in self.eda_summary['target_distribution'].items():
                pct = val / sum(self.eda_summary['target_distribution'].values()) * 100
                label = self.config['target_labels'].get(key, f"Class {key}")
                f.write(f"   {label}: {val} ({pct:.1f}%)\n")
            
            if 'zero_values' in self.eda_summary and self.eda_summary['zero_values']:
                f.write(f"\nZero Values Analysis:\n")
                for feature, stats in self.eda_summary['zero_values'].items():
                    f.write(f"   {feature}: {stats['count']} zeros ({stats['percent']:.1f}%)\n")
            
            if 'outcome_correlations' in self.eda_summary:
                f.write(f"\nTop Correlations with Outcome:\n")
                sorted_corr = sorted(self.eda_summary['outcome_correlations'].items(), 
                                   key=lambda x: abs(x[1]), reverse=True)
                for feature, corr in sorted_corr[1:6]:  # Top 5 (excluding target itself)
                    f.write(f"   {feature}: {corr:.3f}\n")
        
        print(f"[OK] Saved: eda_summary_{self.disease_key}.txt")
        print(f"   Location: {report_path}")
    
    def analyze_dataset(self, disease_key):
        """Execute complete EDA pipeline for a specific disease.
        
        Args:
            disease_key: One of 'diabetes', 'heart_disease', 'stroke'
        """
        print("="*80)
        print(f"STEP 2: EXPLORATORY DATA ANALYSIS - {disease_key.upper()}")
        print("="*80)
        print()
        
        if not self.load_data(disease_key):
            return False
        
        self.basic_statistics()
        self.missing_values_analysis()
        self.correlation_analysis()
        self.distribution_analysis()
        self.outcome_comparison()
        self.pairwise_relationships()
        self.save_summary_report()
        
        print("\n" + "="*80)
        print(f"[OK] STEP 2 COMPLETE: {disease_key.title()} analysis finished")
        print(f"   Visualizations saved to: {self.output_dir}")
        print("="*80 + "\n")
        
        return True
    
    def analyze_all(self):
        """Run EDA for all disease datasets."""
        print("="*80)
        print("STEP 2: EXPLORATORY DATA ANALYSIS - ALL DATASETS")
        print("="*80)
        print()
        
        results = {}
        for disease_key in ['diabetes', 'heart_disease', 'stroke']:
            results[disease_key] = self.analyze_dataset(disease_key)
        
        # Summary
        successful = sum(1 for v in results.values() if v)
        print(f"\n{'='*80}")
        print(f"[OK] Analyzed {successful}/3 datasets successfully")
        print(f"{'='*80}\n")
        
        return results

def main():
    """Execute Step 2: Exploratory Data Analysis"""
    analyzer = ExploratoryAnalysis()
    results = analyzer.analyze_all()
    return results

if __name__ == "__main__":
    main()
