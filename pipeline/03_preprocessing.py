#!/usr/bin/env python3
"""
Step 3: Data Preprocessing
Comprehensive data cleaning, imputation, and feature engineering for disease datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Handles all preprocessing steps for disease datasets."""
    
    def __init__(self):
        self.base_input_dir = Path(__file__).parent / "../datasets/raw"
        self.output_dir = Path(__file__).parent / "../datasets/processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define dataset configurations
        self.dataset_configs = {
            'diabetes': {
                'input_file': 'diabetes.csv',
                'output_file': 'diabetes_processed.csv',
                'target_column': 'Outcome',
                'zero_not_possible': ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'],
                'feature_ranges': {
                    "Glucose": (0, 500),
                    "BloodPressure": (40, 250),
                    "BMI": (10, 60),
                    "Age": (0, 120),
                    "Insulin": (0, 1000),
                    "SkinThickness": (0, 100),
                    "Pregnancies": (0, 20)
                }
            },
            'heart_disease': {
                'input_file': 'heart.csv',
                'output_file': 'heart_disease_processed.csv',
                'target_column': 'target',
                'zero_not_possible': [],
                'feature_ranges': {
                    'age': (18, 100),
                    'trestbps': (80, 220),
                    'chol': (100, 600),
                    'thalach': (60, 220)
                },
                'multi_class_to_binary': True  # Convert 0-4 to 0/1
            },
            'stroke': {
                'input_file': 'healthcare-dataset-stroke-data.csv',
                'output_file': 'stroke_processed.csv',
                'target_column': 'stroke',
                'drop_columns': ['id'],
                'categorical_columns': ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'],
                'zero_not_possible': [],
                'feature_ranges': {
                    'age': (0, 120),
                    'avg_glucose_level': (50, 300),
                    'bmi': (10, 60)
                }
            }
        }
        
        self.dataset = None
        self.processed_data = None
        self.config = None
        self.disease_key = None
    
    def load_data(self, disease_key):
        """Load raw disease dataset.
        
        Args:
            disease_key: One of 'diabetes', 'heart_disease', 'stroke'
        """
        if disease_key not in self.dataset_configs:
            print(f"[X] Unknown disease: {disease_key}")
            return False
        
        self.disease_key = disease_key
        self.config = self.dataset_configs[disease_key]
        input_path = self.base_input_dir / self.config['input_file']
        
        print(f"[>] Loading {disease_key} data from: {input_path}")
        
        try:
            self.dataset = pd.read_csv(input_path)
            
            # Replace '?' with NaN (common in UCI datasets)
            self.dataset = self.dataset.replace('?', np.nan)
            
            # Convert to numeric where possible
            for col in self.dataset.columns:
                if self.dataset[col].dtype == 'object':
                    # Try to convert to numeric if not a categorical column
                    if 'categorical_columns' not in self.config or col not in self.config.get('categorical_columns', []):
                        try:
                            self.dataset[col] = pd.to_numeric(self.dataset[col], errors='coerce')
                        except:
                            pass
            
            print(f"[OK] Loaded dataset: {self.dataset.shape[0]} rows, {self.dataset.shape[1]} columns")
            return True
            
        except Exception as e:
            print(f"[X] Error loading data: {e}")
            return False
    
    def handle_missing_values(self):
        """Handle missing values based on dataset type."""
        print("\n" + "="*80)
        print("HANDLING MISSING VALUES")
        print("="*80)
        
        df = self.dataset.copy()
        initial_rows = len(df)
        
        # Drop specified columns
        if 'drop_columns' in self.config:
            for col in self.config['drop_columns']:
                if col in df.columns:
                    df = df.drop(col, axis=1)
                    print(f"[*] Dropped '{col}' column")
        
        # Encode categorical features
        if 'categorical_columns' in self.config:
            print(f"\n[*] Encoding {len(self.config['categorical_columns'])} categorical features:")
            for col in self.config['categorical_columns']:
                if col in df.columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    print(f"   {col}: encoded")
        
        # Handle zero values that represent missing data
        zero_not_possible = self.config.get('zero_not_possible', [])
        if zero_not_possible:
            print(f"\n[*] Removing rows with zeros in medical features:")
            keep_mask = pd.Series([True] * len(df), index=df.index)
            
            for col in zero_not_possible:
                if col in df.columns:
                    zero_count = (df[col] == 0).sum()
                    if zero_count > 0:
                        zero_pct = (zero_count / len(df)) * 100
                        print(f"   {col}: {zero_count} zeros ({zero_pct:.1f}%) - marking for removal")
                        keep_mask = keep_mask & (df[col] != 0)
            
            df = df[keep_mask].copy()
            rows_removed = initial_rows - len(df)
            
            if rows_removed > 0:
                print(f"\n[OK] Removed {rows_removed} rows with missing values (zeros)")
                print(f"   Dataset size: {initial_rows} -> {len(df)} rows ({len(df)/initial_rows*100:.1f}% retained)")
        
        # Handle any remaining NaN values with median imputation
        remaining_missing = df.isnull().sum().sum()
        if remaining_missing > 0:
            print(f"\n[*] Imputing {remaining_missing} remaining NaN values with median...")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            target_col = self.config['target_column']
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            
            imputer = SimpleImputer(strategy='median')
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        # Convert multi-class target to binary if needed
        target_col = self.config['target_column']
        if self.config.get('multi_class_to_binary', False) and target_col in df.columns:
            original_dist = df[target_col].value_counts().sort_index()
            print(f"\n[*] Converting target from multi-class to binary:")
            print(f"   Original distribution: {dict(original_dist)}")
            df[target_col] = (df[target_col] > 0).astype(int)
            print(f"   Binary distribution: No disease (0): {(df[target_col]==0).sum()}, Disease (1): {(df[target_col]==1).sum()}")
        
        print(f"\n[OK] Data cleaning complete. Final dataset has {len(df)} rows.")
        
        self.processed_data = df
    
    def remove_outliers(self):
        """Remove extreme outliers that are medically impossible."""
        print("\n" + "="*80)
        print("OUTLIER DETECTION & REMOVAL")
        print("="*80)
        
        df = self.processed_data.copy()
        initial_rows = len(df)
        
        # Get feature ranges for this dataset
        feature_ranges = self.config.get('feature_ranges', {})
        
        if not feature_ranges:
            print(f"[*] No outlier removal configured for {self.disease_key}")
            return
        
        print(f"\n[?] Checking for extreme outliers:")
        outliers_removed = 0
        
        for feature, (min_val, max_val) in feature_ranges.items():
            if feature in df.columns:
                outlier_mask = (df[feature] < min_val) | (df[feature] > max_val)
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    print(f"   {feature}: {outlier_count} values outside [{min_val}, {max_val}]")
                    df = df[~outlier_mask]
                    outliers_removed += outlier_count
        
        if outliers_removed > 0:
            print(f"\n[OK] Removed {outliers_removed} outlier records")
            print(f"   Dataset size: {initial_rows} -> {len(df)} rows")
        else:
            print(f"\n[OK] No extreme outliers found")
        
        self.processed_data = df
    
    def balance_imbalanced_data(self):
        """Balance dataset using SMOTE (Synthetic Minority Over-sampling Technique)."""
        if self.processed_data is None:
            print("[X] No processed data available for balancing")
            return
        
        target_col = self.config['target_column']
        if target_col not in self.processed_data.columns:
            print(f"[X] Target column '{target_col}' not found")
            return
        
        df = self.processed_data
        
        print("\n" + "="*80)
        print("CLASS BALANCE ANALYSIS")
        print("="*80)
        
        # Check class distribution
        class_counts = df[target_col].value_counts()
        print(f"\n[*] Original class distribution:")
        for cls, count in class_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   Class {cls}: {count} samples ({percentage:.1f}%)")
        
        # Calculate imbalance ratio
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()
        imbalance_ratio = class_counts[majority_class] / class_counts[minority_class]
        
        print(f"\n[*] Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        # Apply SMOTE if severely imbalanced (ratio > 3:1)
        if imbalance_ratio > 3.0:
            print(f"\n[!] Significant class imbalance detected (>{3}:1)")
            print(f"[>] Applying SMOTE (Synthetic Minority Over-sampling)...")
            
            try:
                from imblearn.over_sampling import SMOTE
                
                # Separate features and target
                X = df.drop(columns=[target_col])
                y = df[target_col]
                
                # Determine k_neighbors - must be less than minority class samples
                k_neighbors = min(5, class_counts[minority_class] - 1)
                
                # Apply SMOTE
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                
                # Combine back into DataFrame
                balanced_df = X_resampled.copy()
                balanced_df[target_col] = y_resampled
                
                # Round categorical columns to nearest integer (SMOTE creates continuous values)
                if 'categorical_columns' in self.config:
                    categorical_cols = self.config['categorical_columns']
                    for col in categorical_cols:
                        if col in balanced_df.columns:
                            # Round to nearest integer and clip to valid range
                            balanced_df[col] = balanced_df[col].round().astype(int)
                            # Ensure values stay within original min/max
                            original_min = df[col].min()
                            original_max = df[col].max()
                            balanced_df[col] = balanced_df[col].clip(original_min, original_max)
                
                # Update processed data
                self.processed_data = balanced_df
                
                # Show new distribution
                new_class_counts = balanced_df[target_col].value_counts()
                print(f"\n[OK] Dataset balanced using SMOTE:")
                print(f"   Original size: {len(df)} samples")
                print(f"   Balanced size: {len(balanced_df)} samples")
                for cls, count in new_class_counts.items():
                    percentage = (count / len(balanced_df)) * 100
                    print(f"   Class {cls}: {count} samples ({percentage:.1f}%)")
                
                new_ratio = new_class_counts[majority_class] / new_class_counts[minority_class]
                print(f"   New balance ratio: {new_ratio:.2f}:1")
                
            except ImportError:
                print("\n[X] ERROR: imbalanced-learn library not installed")
                print("[!] Install with: pip install imbalanced-learn")
                print("[*] Skipping rebalancing...")
            except Exception as e:
                print(f"\n[X] ERROR applying SMOTE: {e}")
                print("[*] Keeping original dataset...")
            
        else:
            print(f"\n[OK] Dataset is reasonably balanced (ratio {imbalance_ratio:.2f}:1)")
            print(f"   No rebalancing needed")
    
    def save_processed_data(self):
        """Save the processed dataset."""
        print("\n" + "="*80)
        print("SAVING PROCESSED DATA")
        print("="*80)
        
        # Save full processed dataset
        output_path = self.output_dir / self.config['output_file']
        self.processed_data.to_csv(output_path, index=False)
        
        print(f"\n[OK] Saved processed dataset:")
        print(f"   Location: {output_path}")
        print(f"   Shape: {self.processed_data.shape}")
        print(f"   Features: {len(self.processed_data.columns) - 1}")
        print(f"   Samples: {len(self.processed_data)}")
        
        # Create preprocessing summary
        summary = {
            "original_rows": len(self.dataset),
            "processed_rows": len(self.processed_data),
            "original_features": len(self.dataset.columns) - 1,
            "processed_features": len(self.processed_data.columns) - 1,
            "columns": list(self.processed_data.columns)
        }
        
        summary_path = self.output_dir / f"preprocessing_summary_{self.disease_key}.txt"
        with open(summary_path, 'w') as f:
            f.write(f"PREPROCESSING SUMMARY - {self.disease_key.upper()}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Original rows: {summary['original_rows']}\n")
            f.write(f"Processed rows: {summary['processed_rows']}\n")
            f.write(f"Original features: {summary['original_features']}\n")
            f.write(f"Processed features: {summary['processed_features']}\n\n")
            f.write("Final Columns:\n")
            for col in summary['columns']:
                f.write(f"   - {col}\n")
        
        print(f"\n[OK] Saved preprocessing summary: {summary_path}")
        
        return output_path
    
    def preprocess_dataset(self, disease_key):
        """Execute complete preprocessing pipeline for a specific disease.
        
        Args:
            disease_key: One of 'diabetes', 'heart_disease', 'stroke'
        """
        print("="*80)
        print(f"STEP 3: DATA PREPROCESSING - {disease_key.upper()}")
        print("="*80)
        print()
        
        if not self.load_data(disease_key):
            return None
        
        self.handle_missing_values()
        self.remove_outliers()
        self.balance_imbalanced_data() 
        output_path = self.save_processed_data()
        
        print("\n" + "="*80)
        print(f"[OK] STEP 3 COMPLETE: {disease_key.title()} preprocessing finished")
        print(f"   Processed data ready for model training")
        print("="*80 + "\n")
        
        return output_path
    
    def preprocess_all(self):
        """Run preprocessing for all disease datasets."""
        print("="*80)
        print("STEP 3: DATA PREPROCESSING - ALL DATASETS")
        print("="*80)
        print()
        
        results = {}
        for disease_key in ['diabetes', 'heart_disease', 'stroke']:
            results[disease_key] = self.preprocess_dataset(disease_key)
        
        # Summary
        successful = sum(1 for v in results.values() if v is not None)
        print(f"\n{'='*80}")
        print(f"[OK] Preprocessed {successful}/3 datasets successfully")
        print(f"{'='*80}\n")
        
        return results

def main():
    """Execute Step 3: Data Preprocessing"""
    preprocessor = DataPreprocessor()
    results = preprocessor.preprocess_all()
    return results

if __name__ == "__main__":
    main()
