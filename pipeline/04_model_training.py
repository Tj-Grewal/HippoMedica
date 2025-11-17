#!/usr/bin/env python3
"""
Step 4: Model Training
Trains multiple machine learning models on preprocessed disease datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Trains and evaluates multiple models on disease datasets."""
    
    def __init__(self):
        self.base_data_dir = Path(__file__).parent / "../datasets/processed"
        self.output_dir = Path(__file__).parent / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define dataset configurations
        self.dataset_configs = {
            'diabetes': {
                'data_file': 'diabetes_processed.csv',
                'target_column': 'Outcome',
                'target_labels': {0: 'No Diabetes', 1: 'Diabetes'}
            },
            'heart_disease': {
                'data_file': 'heart_disease_processed.csv',
                'target_column': 'target',
                'target_labels': {0: 'No Heart Disease', 1: 'Heart Disease'}
            },
            'stroke': {
                'data_file': 'stroke_processed.csv',
                'target_column': 'stroke',
                'target_labels': {0: 'No Stroke', 1: 'Stroke'}
            }
        }
        
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.config = None
        self.disease_key = None
        
        self.models = {}
        self.results = {}
    
    def load_processed_data(self, disease_key):
        """Load preprocessed dataset.
        
        Args:
            disease_key: One of 'diabetes', 'heart_disease', 'stroke'
        """
        if disease_key not in self.dataset_configs:
            print(f"[X] Unknown disease: {disease_key}")
            return False
        
        self.disease_key = disease_key
        self.config = self.dataset_configs[disease_key]
        data_path = self.base_data_dir / self.config['data_file']
        
        print(f"[>] Loading {disease_key} data from: {data_path}")
        
        try:
            self.dataset = pd.read_csv(data_path)
            
            target_col = self.config['target_column']
            print(f"[OK] Loaded dataset: {self.dataset.shape[0]} rows, {self.dataset.shape[1]} columns")
            print(f"   Features: {len(self.dataset.columns) - 1}")
            print(f"   Target: {target_col}")
            
            return True
            
        except Exception as e:
            print(f"[X] Error loading data: {e}")
            return False
    
    def prepare_train_test_split(self):
        """Split data into training and testing sets."""
        print("\n" + "="*80)
        print("PREPARING TRAIN-TEST SPLIT")
        print("="*80)
        
        # Separate features and target
        target_col = self.config['target_column']
        X = self.dataset.drop(columns=[target_col])
        y = self.dataset[target_col]
        
        target_labels = self.config['target_labels']
        
        print(f"\n[#] Dataset Summary:")
        print(f"   Total samples: {len(X):,}")
        print(f"   Features: {len(X.columns)}")
        for class_val in sorted(y.unique()):
            label = target_labels.get(class_val, f"Class {class_val}")
            count = (y == class_val).sum()
            pct = count / len(y) * 100
            print(f"   {label}: {count} ({pct:.1f}%)")
        
        # Split with stratification to maintain class balance
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n[>] Data Split (80/20 with stratification):")
        print(f"   Training set: {len(self.X_train):,} samples")
        print(f"   Testing set: {len(self.X_test):,} samples")
        
        # Scale features
        print(f"\n[*] Scaling features using StandardScaler...")
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"[OK] Data preparation complete")
    
    def initialize_models(self):
        """Initialize machine learning models."""
        print("\n" + "="*80)
        print("INITIALIZING MODELS")
        print("="*80)
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=None,  # Allow trees to grow deeper
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',  # Handle class imbalance
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=200,
                max_depth=8, 
                learning_rate=0.05,  # Lower learning rate for better learning
                scale_pos_weight=2.0,  # Handle class imbalance (ratio of neg/pos)
                random_state=42,
                eval_metric='logloss'
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                max_iter=1000,
                random_state=42,
                early_stopping=True
            )
        }
        
        print(f"\n[AI] Initialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"   [+] {name}")
    
    def train_and_evaluate_model(self, name, model, use_scaled=False):
        """Train and evaluate a single model."""
        print(f"\n[-] Training {name}...")
        
        # Select appropriate data (scaled for Neural Network, unscaled for tree-based)
        if use_scaled:
            X_train_data = self.X_train_scaled
            X_test_data = self.X_test_scaled
        else:
            X_train_data = self.X_train
            X_test_data = self.X_test
        
        # Train model
        model.fit(X_train_data, self.y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_data)
        y_pred_proba = model.predict_proba(X_test_data)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_data, self.y_train, 
                                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                    scoring='accuracy')
        
        # Store results
        self.results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'feature_names': list(self.X_train.columns)
        }
        
        # Print results
        print(f"   [OK] {name} Results:")
        print(f"      Accuracy:     {accuracy:.4f}")
        print(f"      Precision:    {precision:.4f}")
        print(f"      Recall:       {recall:.4f}")
        print(f"      F1-Score:     {f1:.4f}")
        print(f"      ROC-AUC:      {roc_auc:.4f}")
        print(f"      CV Accuracy:  {cv_scores.mean():.4f} ({cv_scores.std():.4f})")
        
        # Confusion matrix breakdown
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"      Sensitivity:  {sensitivity:.4f}")
        print(f"      Specificity:  {specificity:.4f}")
    
    def train_all_models(self):
        """Train and evaluate all models."""
        print("\n" + "="*80)
        print("TRAINING ALL MODELS")
        print("="*80)
        
        # Train Random Forest (unscaled)
        self.train_and_evaluate_model('Random Forest', self.models['Random Forest'], use_scaled=False)
        
        # Train XGBoost (unscaled)
        self.train_and_evaluate_model('XGBoost', self.models['XGBoost'], use_scaled=False)
        
        # Train Neural Network (scaled)
        self.train_and_evaluate_model('Neural Network', self.models['Neural Network'], use_scaled=True)
    
    def print_model_comparison(self):
        """Print comparison of all models."""
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)
        
        print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
        print("-"*92)
        
        for name in ['Random Forest', 'XGBoost', 'Neural Network']:
            if name in self.results:
                r = self.results[name]
                print(f"{name:<20} {r['accuracy']:<12.4f} {r['precision']:<12.4f} "
                      f"{r['recall']:<12.4f} {r['f1_score']:<12.4f} {r['roc_auc']:<12.4f}")
        
        # Find best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_accuracy = self.results[best_model_name]['accuracy']
        
        print("\n" + "="*80)
        print(f"[!] BEST MODEL: {best_model_name}")
        print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        print("="*80)
    
    def save_training_summary(self):
        """Save training results summary."""
        print("\n" + "="*80)
        print("SAVING TRAINING SUMMARY")
        print("="*80)
        
        summary_path = self.output_dir / f"training_summary_{self.disease_key}.txt"
        
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"MODEL TRAINING SUMMARY - {self.disease_key.upper()}\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Dataset: {self.config['data_file']}\n")
            f.write(f"Training samples: {len(self.X_train)}\n")
            f.write(f"Testing samples: {len(self.X_test)}\n")
            f.write(f"Features: {len(self.X_train.columns)}\n\n")
            
            f.write("Model Performance:\n")
            f.write("-"*80 + "\n")
            
            for name in ['Random Forest', 'XGBoost', 'Neural Network']:
                if name in self.results:
                    r = self.results[name]
                    f.write(f"\n{name}:\n")
                    f.write(f"  Accuracy:     {r['accuracy']:.4f}\n")
                    f.write(f"  Precision:    {r['precision']:.4f}\n")
                    f.write(f"  Recall:       {r['recall']:.4f}\n")
                    f.write(f"  F1-Score:     {r['f1_score']:.4f}\n")
                    f.write(f"  ROC-AUC:      {r['roc_auc']:.4f}\n")
                    f.write(f"  CV Accuracy:  {r['cv_mean']:.4f} ({r['cv_std']:.4f})\n")
                    
                    tn, fp, fn, tp = r['confusion_matrix'].ravel()
                    f.write(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}\n")
            
            # Best model
            best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
            f.write(f"\nBest Model: {best_model_name}\n")
            f.write(f"Accuracy: {self.results[best_model_name]['accuracy']:.4f}\n")
        
        print(f"[OK] Saved training summary: {summary_path}")
    
    def train_models(self, disease_key):
        """Execute complete model training pipeline for a specific disease.
        
        Args:
            disease_key: One of 'diabetes', 'heart_disease', 'stroke'
        """
        print("="*80)
        print(f"STEP 4: MODEL TRAINING - {disease_key.upper()}")
        print("="*80)
        print()
        
        # Reset instance state for each disease
        self.results = {}
        self.models = {}
        
        if not self.load_processed_data(disease_key):
            return None
        
        self.prepare_train_test_split()
        self.initialize_models()
        self.train_all_models()
        self.print_model_comparison()
        self.save_training_summary()
        
        print("\n" + "="*80)
        print(f"[OK] STEP 4 COMPLETE: {disease_key.title()} training finished")
        print(f"   Trained {len(self.results)} models successfully")
        print("="*80 + "\n")
        
        # Return a COPY of results to prevent reference issues
        return {
            'results': dict(self.results),
            'scaler': self.scaler
        }
    
    def train_all(self):
        """Train models for all disease datasets."""
        print("="*80)
        print("STEP 4: MODEL TRAINING - ALL DATASETS")
        print("="*80)
        print()
        
        all_results = {}
        for disease_key in ['diabetes', 'heart_disease', 'stroke']:
            result = self.train_models(disease_key)
            all_results[disease_key] = result
        
        # Summary
        successful = sum(1 for v in all_results.values() if v is not None)
        print(f"\n{'='*80}")
        print(f"[OK] Trained models for {successful}/3 datasets successfully")
        print(f"{'='*80}\n")
        
        return all_results

def main():
    """Execute Step 4: Model Training"""
    trainer = ModelTrainer()
    results = trainer.train_all()
    return results

if __name__ == "__main__":
    main()
