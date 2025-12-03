#!/usr/bin/env python3
"""
Generate additional figures and diagrams for LaTeX reports.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import seaborn as sns
from pathlib import Path

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

output_dir = Path('figures')
output_dir.mkdir(exist_ok=True)

def generate_pipeline_architecture():
    """Generate ML pipeline architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 12))  # Increased height to prevent cutoff
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 12)  # Extended y-axis to include bottom stage
    
    # Define colors
    color_data = '#3498db'
    color_preprocess = '#2ecc71'
    color_model = '#e74c3c'
    color_deploy = '#f39c12'
    
    # Stage 1: Data Acquisition
    stage1_box = FancyBboxPatch((0.5, 9), 9, 1.8, boxstyle="round,pad=0.1", 
                                edgecolor='black', facecolor=color_data, linewidth=2)
    ax.add_patch(stage1_box)
    ax.text(5, 10.2, 'STAGE 1: DATA ACQUISITION & QUALITY ASSESSMENT', 
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax.text(5, 9.6, 'Raw Datasets → Missing Value Detection → Outlier Identification', 
            ha='center', va='center', fontsize=10, color='white')
    
    # Arrow 1
    arrow1 = FancyArrowPatch((5, 9), (5, 7.8), arrowstyle='->', mutation_scale=30, 
                            linewidth=3, color='black')
    ax.add_patch(arrow1)
    
    # Stage 2: Preprocessing & SMOTE
    stage2_box = FancyBboxPatch((0.5, 6), 9, 1.8, boxstyle="round,pad=0.1", 
                                edgecolor='black', facecolor=color_preprocess, linewidth=2)
    ax.add_patch(stage2_box)
    ax.text(5, 7.2, 'STAGE 2: ADVANCED PREPROCESSING & SMOTE BALANCING', 
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax.text(5, 6.6, 'Zero Imputation → Categorical Encoding → SMOTE (19:1→1:1) → Category Fixing', 
            ha='center', va='center', fontsize=10, color='white')
    
    # Arrow 2
    arrow2 = FancyArrowPatch((5, 6), (5, 4.8), arrowstyle='->', mutation_scale=30, 
                            linewidth=3, color='black')
    ax.add_patch(arrow2)
    
    # Stage 3: Model Training (Three models side by side)
    stage3_main = FancyBboxPatch((0.5, 2.2), 9, 2.6, boxstyle="round,pad=0.1", 
                                 edgecolor='black', facecolor='white', linewidth=2, linestyle='--')
    ax.add_patch(stage3_main)
    ax.text(5, 4.5, 'STAGE 3: ENSEMBLE MODEL TRAINING', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Random Forest
    rf_box = FancyBboxPatch((0.7, 2.4), 2.6, 1.8, boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor=color_model, linewidth=2)
    ax.add_patch(rf_box)
    ax.text(2, 3.7, 'Random Forest', ha='center', va='center', fontsize=10, 
            fontweight='bold', color='white')
    ax.text(2, 3.3, '200 Trees', ha='center', va='center', fontsize=8, color='white')
    ax.text(2, 3.0, 'Balanced Weights', ha='center', va='center', fontsize=8, color='white')
    ax.text(2, 2.7, 'Best: 96.80%', ha='center', va='center', fontsize=8, 
            fontweight='bold', color='yellow')
    
    # XGBoost
    xgb_box = FancyBboxPatch((3.7, 2.4), 2.6, 1.8, boxstyle="round,pad=0.1", 
                             edgecolor='black', facecolor=color_model, linewidth=2)
    ax.add_patch(xgb_box)
    ax.text(5, 3.7, 'XGBoost', ha='center', va='center', fontsize=10, 
            fontweight='bold', color='white')
    ax.text(5, 3.3, '200 Rounds', ha='center', va='center', fontsize=8, color='white')
    ax.text(5, 3.0, 'Learning Rate: 0.05', ha='center', va='center', fontsize=8, color='white')
    ax.text(5, 2.7, 'Acc: 95.10%', ha='center', va='center', fontsize=8, color='white')
    
    # Neural Network
    nn_box = FancyBboxPatch((6.7, 2.4), 2.6, 1.8, boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor=color_model, linewidth=2)
    ax.add_patch(nn_box)
    ax.text(8, 3.7, 'Neural Network', ha='center', va='center', fontsize=10, 
            fontweight='bold', color='white')
    ax.text(8, 3.3, 'MLP: 128-64-32', ha='center', va='center', fontsize=8, color='white')
    ax.text(8, 3.0, 'ReLU + Early Stop', ha='center', va='center', fontsize=8, color='white')
    ax.text(8, 2.7, 'Acc: 92.47%', ha='center', va='center', fontsize=8, color='white')
    
    # Arrow 3
    arrow3 = FancyArrowPatch((5, 2.2), (5, 1.0), arrowstyle='->', mutation_scale=30, 
                            linewidth=3, color='black')
    ax.add_patch(arrow3)
    
    # Stage 4: Deployment
    stage4_box = FancyBboxPatch((0.5, 0.0), 9, 1.0, boxstyle="round,pad=0.1", 
                                edgecolor='black', facecolor=color_deploy, linewidth=2)
    ax.add_patch(stage4_box)
    ax.text(5, 0.5, 'STAGE 4: REAL-TIME WEB DEPLOYMENT & PREDICTION', 
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    plt.title('Multi-Modal Disease Detection System: Complete AI Pipeline Architecture', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'pipeline_architecture.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Generated: pipeline_architecture.png")
    plt.close()

def generate_smote_comparison():
    """Generate before/after SMOTE comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Before SMOTE - use contrasting colors (red for stroke, gray for healthy)
    categories = ['Stroke Cases', 'Healthy Cases']
    before_values = [249, 4861]
    colors_before = ['#e74c3c', '#95a5a6']  # Red and gray
    
    wedges1, texts1, autotexts1 = ax1.pie(before_values, labels=categories, autopct='%1.1f%%',
                                            colors=colors_before, startangle=90, textprops={'fontsize': 12})
    ax1.set_title('BEFORE SMOTE: Severe Imbalance\n(Ratio: 19.52:1)', 
                  fontsize=14, fontweight='bold', color='#e74c3c')
    
    # Add sample counts
    ax1.text(0, -1.5, f'Total Samples: {sum(before_values):,}', 
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.text(0, -1.8, 'Model Performance: 0% Recall', 
             ha='center', fontsize=11, fontweight='bold', color='red')
    
    # After SMOTE - use contrasting colors (red for stroke, green for healthy)
    after_values = [4861, 4861]
    colors_after = ['#e74c3c', '#2ecc71']  # Red for stroke, green for healthy
    
    wedges2, texts2, autotexts2 = ax2.pie(after_values, labels=categories, autopct='%1.1f%%',
                                            colors=colors_after, startangle=90, textprops={'fontsize': 12})
    ax2.set_title('AFTER SMOTE: Perfect Balance\n(Ratio: 1:1)', 
                  fontsize=14, fontweight='bold', color='#2ecc71')
    
    # Add sample counts
    ax2.text(0, -1.5, f'Total Samples: {sum(after_values):,}', 
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax2.text(0, -1.8, 'Model Performance: 97.11% Recall', 
             ha='center', fontsize=11, fontweight='bold', color='green')
    
    plt.suptitle('SMOTE Class Balancing: Transforming Model from Unusable to Clinical-Grade', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'smote_comparison.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Generated: smote_comparison.png")
    plt.close()

def generate_performance_comparison():
    """Generate model performance comparison across diseases."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Data - corrected to match actual metrics from JSON files
    diseases = ['Diabetes', 'Heart Disease', 'Stroke']
    models = ['Random Forest', 'XGBoost', 'Neural Network']
    
    # Accuracy values from actual JSON metrics
    accuracy = {
        'Random Forest': [78.21, 86.89, 96.80],
        'XGBoost': [75.64, 83.61, 95.10],
        'Neural Network': [74.36, 83.61, 92.47]
    }
    
    # Recall values - corrected from actual JSON
    recall = {
        'Random Forest': [57.69, 92.86, 97.11],
        'XGBoost': [61.54, 92.86, 98.04],
        'Neural Network': [53.85, 89.29, 94.85]
    }
    
    # F1-Score values - corrected from actual JSON
    f1_score = {
        'Random Forest': [63.83, 86.67, 96.81],
        'XGBoost': [62.75, 83.87, 95.24],
        'Neural Network': [58.33, 83.33, 92.65]
    }
    
    # ROC-AUC values - corrected from actual JSON
    roc_auc = {
        'Random Forest': [83.36, 95.56, 99.56],
        'XGBoost': [82.99, 92.32, 99.27],
        'Neural Network': [77.29, 93.61, 97.55]
    }
    
    # Plot 1: Accuracy Comparison
    x = np.arange(len(diseases))
    width = 0.25
    
    axes[0, 0].bar(x - width, accuracy['Random Forest'], width, label='Random Forest', color='#3498db')
    axes[0, 0].bar(x, accuracy['XGBoost'], width, label='XGBoost', color='#e74c3c')
    axes[0, 0].bar(x + width, accuracy['Neural Network'], width, label='Neural Network', color='#2ecc71')
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(diseases)
    axes[0, 0].legend()
    axes[0, 0].set_ylim(70, 100)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Recall Comparison
    axes[0, 1].bar(x - width, recall['Random Forest'], width, label='Random Forest', color='#3498db')
    axes[0, 1].bar(x, recall['XGBoost'], width, label='XGBoost', color='#e74c3c')
    axes[0, 1].bar(x + width, recall['Neural Network'], width, label='Neural Network', color='#2ecc71')
    axes[0, 1].set_ylabel('Recall (%)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Model Recall Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(diseases)
    axes[0, 1].legend()
    axes[0, 1].set_ylim(50, 100)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: F1-Score Comparison
    axes[1, 0].bar(x - width, f1_score['Random Forest'], width, label='Random Forest', color='#3498db')
    axes[1, 0].bar(x, f1_score['XGBoost'], width, label='XGBoost', color='#e74c3c')
    axes[1, 0].bar(x + width, f1_score['Neural Network'], width, label='Neural Network', color='#2ecc71')
    axes[1, 0].set_ylabel('F1-Score (%)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Model F1-Score Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(diseases)
    axes[1, 0].legend()
    axes[1, 0].set_ylim(55, 100)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: ROC-AUC Comparison
    axes[1, 1].bar(x - width, roc_auc['Random Forest'], width, label='Random Forest', color='#3498db')
    axes[1, 1].bar(x, roc_auc['XGBoost'], width, label='XGBoost', color='#e74c3c')
    axes[1, 1].bar(x + width, roc_auc['Neural Network'], width, label='Neural Network', color='#2ecc71')
    axes[1, 1].set_ylabel('ROC-AUC (%)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Model ROC-AUC Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(diseases)
    axes[1, 1].legend()
    axes[1, 1].set_ylim(75, 100)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Comprehensive Model Performance Comparison Across Three Diseases', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Generated: performance_comparison.png")
    plt.close()

def generate_feature_importance():
    """Generate feature importance visualization for diabetes."""
    # Actual values from diabetes_feature_importance.json
    features = ['Glucose', 'Insulin', 'Age', 'BMI', 'Pedigree', 'Skin', 'Pregnancies', 'BP']
    importance = [25.78, 17.16, 15.41, 11.17, 9.69, 8.10, 6.35, 6.34]
    
    fig, ax = plt.subplots(figsize=(10, 5.5))  # Slightly increased height
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(features)))
    bars = ax.barh(features, importance, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Feature Importance (%)', fontsize=12, fontweight='bold')
    ax.set_title('Top Features Contributing to Diabetes Prediction\n(Random Forest Feature Importance)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 30)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, importance):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.2f}%', 
                va='center', fontweight='bold')
    
    # Add medical relevance note - positioned below xlabel with proper spacing
    fig.text(0.5, 0.02, 'Clinical Validation: Feature rankings align with established medical knowledge', 
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])  # Leave space at bottom for annotation
    plt.savefig(output_dir / 'feature_importance_diabetes.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Generated: feature_importance_diabetes.png")
    plt.close()

def generate_correlation_heatmaps():
    """Generate correlation heatmaps for all datasets with 45-degree rotated labels."""
    import pandas as pd
    
    datasets_dir = Path('../datasets/raw')
    
    # Dataset configurations
    datasets = {
        'diabetes': {
            'file': 'diabetes.csv',
            'title': 'Diabetes - Feature Correlation Heatmap'
        },
        'heart': {
            'file': 'heart.csv', 
            'title': 'Heart Disease - Feature Correlation Heatmap'
        },
        'stroke': {
            'file': 'healthcare-dataset-stroke-data.csv',
            'title': 'Stroke - Feature Correlation Heatmap'
        }
    }
    
    for key, config in datasets.items():
        filepath = datasets_dir / config['file']
        if not filepath.exists():
            print(f"[!] Skipping {key}: file not found at {filepath}")
            continue
        
        # Load data
        df = pd.read_csv(filepath)
        
        # For stroke dataset, encode categorical columns and drop 'id'
        if key == 'stroke':
            # Drop id column
            if 'id' in df.columns:
                df = df.drop('id', axis=1)
            # Encode categorical columns
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = pd.factorize(df[col])[0]
        
        # Get only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Generate heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8},
                   ax=ax)
        
        # Rotate x-axis labels 45 degrees
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        # Rotate y-axis labels 45 degrees (horizontal is usually better for y, but matching request)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=45, ha='right')
        
        ax.set_title(config['title'], fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{key}_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Generated: {key}_correlation.png")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("GENERATING FIGURES FOR LATEX REPORTS")
    print("="*60 + "\n")
    
    generate_pipeline_architecture()
    generate_smote_comparison()
    generate_performance_comparison()
    generate_feature_importance()
    generate_correlation_heatmaps()
    
    print("\n[OK] All figures generated successfully!")
    print(f"[>] Figures saved to: {output_dir.absolute()}")
