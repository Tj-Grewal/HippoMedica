#!/usr/bin/env python3
"""
Step 5: Model Storage
Saves trained models, scaler, and performance metrics for deployment.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelStorage:
    """Handles saving models, scalers, and metadata."""
    
    def __init__(self, models_dir="../models"):
        self.models_dir = Path(__file__).parent / models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.models_dir / "trained_models").mkdir(exist_ok=True)
        (self.models_dir / "scalers").mkdir(exist_ok=True)
        (self.models_dir / "metadata").mkdir(exist_ok=True)
        
        self.disease_key = None
        self.disease_key = None
    
    def save_models_for_disease(self, disease_key, model_results):
        """Save all trained models for a specific disease using joblib.
        
        Args:
            disease_key: One of 'diabetes', 'heart_disease', 'stroke'
            model_results: Dictionary containing model results
        """
        print(f"\n[*] Saving {disease_key} models...")
        
        saved_models = []
        
        for model_name in ['Random Forest', 'XGBoost', 'Neural Network']:
            if model_name in model_results:
                # Get model
                model = model_results[model_name]['model']
                
                # Create safe filename with disease prefix
                safe_name = model_name.lower().replace(' ', '_')
                model_path = self.models_dir / "trained_models" / f"{disease_key}_{safe_name}.pkl"
                
                # Save model
                joblib.dump(model, model_path)
                saved_models.append(model_name)
                
                print(f"   [+] Saved {model_name} → {model_path.name}")
        
        print(f"[OK] Saved {len(saved_models)} {disease_key} models")
        return saved_models
    
    def save_scaler_for_disease(self, disease_key, scaler, feature_names):
        """Save the StandardScaler for a specific disease.
        
        Args:
            disease_key: One of 'diabetes', 'heart_disease', 'stroke'
            scaler: The fitted StandardScaler
            feature_names: List of feature names
        """
        print(f"\n[*] Saving {disease_key} scaler...")
        
        scaler_path = self.models_dir / "scalers" / f"{disease_key}_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        
        # Save feature names for reference
        features_path = self.models_dir / "scalers" / f"{disease_key}_features.txt"
        with open(features_path, 'w') as f:
            f.write(f"Feature Names for {disease_key.upper()} (in order):\n")
            f.write("="*50 + "\n")
            for idx, feature in enumerate(feature_names, 1):
                f.write(f"{idx}. {feature}\n")
        
        print(f"   [+] Saved scaler → {scaler_path.name}")
        print(f"   [+] Saved feature list → {features_path.name}")
        print(f"   [+] Saved scaler → {scaler_path.name}")
        print(f"   [+] Saved feature list → {features_path.name}")
    
    def save_performance_metrics_for_disease(self, disease_key, model_results):
        """Save detailed performance metrics for a specific disease.
        
        Args:
            disease_key: One of 'diabetes', 'heart_disease', 'stroke'
            model_results: Dictionary containing model results
        """
        print(f"\n[*] Saving {disease_key} performance metrics...")
        
        metrics_data = {}
        
        for model_name in ['Random Forest', 'XGBoost', 'Neural Network']:
            if model_name in model_results:
                r = model_results[model_name]
                
                # Extract metrics (convert numpy types to native Python types)
                cm = r['confusion_matrix']
                tn, fp, fn, tp = cm.ravel()
                
                metrics_data[model_name] = {
                    'accuracy': float(r['accuracy']),
                    'precision': float(r['precision']),
                    'recall': float(r['recall']),
                    'f1_score': float(r['f1_score']),
                    'roc_auc': float(r['roc_auc']),
                    'cv_mean': float(r['cv_mean']),
                    'cv_std': float(r['cv_std']),
                    'confusion_matrix': {
                        'true_negatives': int(tn),
                        'false_positives': int(fp),
                        'false_negatives': int(fn),
                        'true_positives': int(tp)
                    },
                    'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                    'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
                }
        
        # Save as JSON
        metrics_path = self.models_dir / "metadata" / f"{disease_key}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"   [+] Saved metrics JSON → {metrics_path.name}")
        
        # Also save as readable text
        text_path = self.models_dir / "metadata" / f"{disease_key}_summary.txt"
        with open(text_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"MODEL PERFORMANCE SUMMARY - {disease_key.upper()}\n")
            f.write("="*80 + "\n\n")
            
            for model_name, metrics in metrics_data.items():
                f.write(f"{model_name}:\n")
                f.write("-"*50 + "\n")
                f.write(f"  Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
                f.write(f"  Precision:    {metrics['precision']:.4f}\n")
                f.write(f"  Recall:       {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score:     {metrics['f1_score']:.4f}\n")
                f.write(f"  ROC-AUC:      {metrics['roc_auc']:.4f}\n")
                f.write(f"  CV Accuracy:  {metrics['cv_mean']:.4f} ({metrics['cv_std']:.4f})\n")
                f.write(f"  Sensitivity:  {metrics['sensitivity']:.4f}\n")
                f.write(f"  Specificity:  {metrics['specificity']:.4f}\n")
                f.write(f"\n  Confusion Matrix:\n")
                cm = metrics['confusion_matrix']
                f.write(f"    True Negatives:  {cm['true_negatives']}\n")
                f.write(f"    False Positives: {cm['false_positives']}\n")
                f.write(f"    False Negatives: {cm['false_negatives']}\n")
                f.write(f"    True Positives:  {cm['true_positives']}\n")
                f.write("\n")
            
            # Best model
            best_model = max(metrics_data.keys(), key=lambda x: metrics_data[x]['accuracy'])
            f.write("="*80 + "\n")
            f.write(f"BEST MODEL: {best_model}\n")
            f.write(f"Accuracy: {metrics_data[best_model]['accuracy']:.4f}\n")
            f.write("="*80 + "\n")
        
        print(f"   [+] Saved readable summary → {text_path.name}")
        print(f"   [+] Saved readable summary → {text_path.name}")
    
    def save_feature_importance_for_disease(self, disease_key, model_results):
        """Save feature importance for tree-based models.
        
        Args:
            disease_key: One of 'diabetes', 'heart_disease', 'stroke'
            model_results: Dictionary containing model results
        """
        print(f"\n[*] Saving {disease_key} feature importance...")
        
        feature_importance_data = {}
        
        for model_name in ['Random Forest', 'XGBoost']:
            if model_name in model_results:
                model = model_results[model_name]['model']
                feature_names = model_results[model_name]['feature_names']
                
                # Get feature importances
                importances = model.feature_importances_
                
                # Create sorted list
                importance_list = [
                    {'feature': name, 'importance': float(imp)}
                    for name, imp in zip(feature_names, importances)
                ]
                importance_list.sort(key=lambda x: x['importance'], reverse=True)
                
                feature_importance_data[model_name] = importance_list
        
        # Save as JSON
        importance_path = self.models_dir / "metadata" / f"{disease_key}_feature_importance.json"
        with open(importance_path, 'w') as f:
            json.dump(feature_importance_data, f, indent=2)
        
        print(f"   [+] Saved feature importance → {importance_path.name}")
        
        # Save readable text version
        text_path = self.models_dir / "metadata" / f"{disease_key}_feature_importance.txt"
        with open(text_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"FEATURE IMPORTANCE ANALYSIS - {disease_key.upper()}\n")
            f.write("="*80 + "\n\n")
            
            for model_name, importance_list in feature_importance_data.items():
                f.write(f"{model_name}:\n")
                f.write("-"*50 + "\n")
                for idx, item in enumerate(importance_list[:10], 1):  # Top 10
                    f.write(f"  {idx:2d}. {item['feature']:<30s} {item['importance']:.6f}\n")
                f.write("\n")
        
        print(f"   [+] Saved readable importance → {text_path.name}")
        print(f"   [+] Saved readable importance → {text_path.name}")
    
    def save_models(self, disease_key, results_dict):
        """Save models, scaler, and metadata for a specific disease.
        
        Args:
            disease_key: One of 'diabetes', 'heart_disease', 'stroke'
            results_dict: Dictionary containing 'results' and 'scaler' keys
        """
        print("="*80)
        print(f"STEP 5: MODEL STORAGE - {disease_key.upper()}")
        print("="*80)
        
        if results_dict is None:
            print(f"[X] No results provided for {disease_key}")
            return None
        
        model_results = results_dict['results']
        scaler = results_dict['scaler']
        feature_names = model_results[list(model_results.keys())[0]]['feature_names']
        
        # Save all components
        self.save_models_for_disease(disease_key, model_results)
        self.save_scaler_for_disease(disease_key, scaler, feature_names)
        self.save_performance_metrics_for_disease(disease_key, model_results)
        self.save_feature_importance_for_disease(disease_key, model_results)
        
        # Find best model
        best_model_name = max(model_results.keys(), 
                             key=lambda x: model_results[x]['accuracy'])
        
        print("\n" + "="*80)
        print(f"[OK] STEP 5 COMPLETE: {disease_key.title()} models saved")
        print(f"   Location: {self.models_dir}")
        print(f"   Best model: {best_model_name}")
        print("="*80 + "\n")
        
        return self.models_dir
    
    def save_all(self, all_results):
        """Save models for all diseases.
        
        Args:
            all_results: Dictionary mapping disease_key to results_dict
        """
        print("="*80)
        print("STEP 5: MODEL STORAGE - ALL DATASETS")
        print("="*80)
        print()
        
        saved = {}
        for disease_key in ['diabetes', 'heart_disease', 'stroke']:
            if disease_key in all_results and all_results[disease_key] is not None:
                saved[disease_key] = self.save_models(disease_key, all_results[disease_key])
        
        # Summary
        successful = len(saved)
        print(f"\n{'='*80}")
        print(f"[OK] Saved models for {successful}/3 datasets successfully")
        print(f"   Location: {self.models_dir}")
        print(f"{'='*80}\n")
        
        return saved

def main():
    """Execute Step 5: Model Storage"""
    print("[!] This script is designed to be called from run_pipeline.py")
    print("   Please run the complete pipeline instead.")

if __name__ == "__main__":
    main()
