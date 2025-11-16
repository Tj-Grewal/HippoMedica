"""
Random Forest Model for Diabetes Prediction
Using Scikit-Learn with Hyperparameter Tuning and Feature Importance
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


######################################################################
# PREPROCESSING (Same as KNN)
######################################################################

def load_and_preprocess_data(filepath='../datasets/diabetes.csv'):
    """
    Load and preprocess diabetes dataset
    
    Returns:
        X: preprocessed features (n_samples, n_features)
        y: labels (n_samples,)
        scaler: fitted StandardScaler
        feature_names: list of feature names
    """
    print("="*70)
    print("PREPROCESSING DIABETES DATA")
    print("="*70)
    
    # Step 1: Load data with explicit column names
    print("\n1. Loading data")
    column_names = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]
    
    df = pd.read_csv(filepath, names=column_names, header=None)
    print(f"  Loaded {len(df)} samples with {len(df.columns)-1} features")
    
    # Step 2: Drop rows where SkinThickness == 0 OR Insulin == 0, then use ALL 8 features
    print("\n2. Dropping rows where SkinThickness==0 OR Insulin==0, then selecting ALL 8 features")
    before = len(df)
    df = df[(df['SkinThickness'] != 0) & (df['Insulin'] != 0)].copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} samples with zero SkinThickness/Insulin")

    features_to_use = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]

    X = df[features_to_use].copy()
    y = df['Outcome'].values

    print(f"  Selected features: {features_to_use}")
    print(f"  Shape: {X.shape}")

    # Step 3: Handle missing values (zeros in medical features)
    print("\n3. Handling missing values")
    cannot_be_zero = ['Glucose', 'BloodPressure', 'BMI']
    
    for col in cannot_be_zero:
        zero_count = (X[col] == 0).sum()
        if zero_count > 0:
            median_val = X.loc[X[col] != 0, col].median()
            X.loc[X[col] == 0, col] = median_val
            print(f"  Imputed {zero_count} missing values in {col} with median={median_val:.2f}")
    
    # Step 4: Standardize features (except Pregnancies)
    print("\n4. Standardizing features")
    scaler = StandardScaler()

    features_to_standardize = [
        'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]

    X[features_to_standardize] = scaler.fit_transform(X[features_to_standardize])

    print(f"  Standardized {len(features_to_standardize)} features")
    print(f"  Pregnancies kept as raw count")
    
    # Convert to numpy array
    X = X.values
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    
    return X, y, scaler, features_to_use


######################################################################
# HYPERPARAMETER TUNING
######################################################################

def tune_hyperparameters(X_train, y_train, cv=5):
    """
    Find optimal Random Forest hyperparameters using GridSearchCV
    
    Args:
        X_train: training features
        y_train: training labels
        cv: number of CV folds
        
    Returns:
        best_params: dict of optimal parameters
        grid_search: fitted GridSearchCV object
    """
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING WITH GRID SEARCH")
    print("="*70)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200, 300],    # number of trees
        'max_depth': [None, 10, 20, 30],    # maximum depth of each tree
        'min_samples_split': [2, 5, 10],    # minimum samples required to split a node
        'min_samples_leaf': [1, 2, 4],      # minimum samples at a leaf node
        'max_features': ['sqrt', 'log2']    # number of features to consider at each split (2 approaches)
    }
    
    print(f"\nParameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"\nTotal combinations: {total_combinations}")
    print(f"Using {cv}-fold cross-validation")
    print(f"Total fits: {total_combinations * cv}")
    
    print("\nSearching for optimal parameters...")
    # print("(This may take a few minutes)\n")
    
    # Initialize GridSearchCV
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    print("\n" + "="*70)
    print("GRID SEARCH COMPLETE")
    print("="*70)
    
    print(f"\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest CV accuracy: {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.2f}%)")
    
    # Show top 5 parameter combinations
    print(f"\nTop 5 parameter combinations:")
    results_df = pd.DataFrame(grid_search.cv_results_)
    top_5 = results_df.nlargest(5, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
    
    for idx, row in top_5.iterrows():
        print(f"\n  Rank {top_5.index.get_loc(idx) + 1}:")
        print(f"    Accuracy: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
        print(f"    Parameters: {row['params']}")
    
    return grid_search.best_params_, grid_search


######################################################################
# FEATURE IMPORTANCE ANALYSIS
######################################################################

def analyze_feature_importance(model, feature_names, save_plot=True):
    """
    Analyze and visualize feature importance
    
    Args:
        model: trained RandomForestClassifier
        feature_names: list of feature names
        save_plot: whether to save the plot
        
    Returns:
        importance_df: DataFrame with feature importances
    """
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices],
        'Percentage': importances[indices] * 100
    })
    
    print("\nFeature Importance Ranking (Usefulness of features to make predictions):")
    print("="*70)
    for idx, row in importance_df.iterrows():
        print(f"{idx+1}. {row['Feature']:25s} {row['Importance']:.4f} ({row['Percentage']:5.2f}%)")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))
    bars = plt.barh(range(len(importances)), importances[indices], color=colors)
    plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, imp) in enumerate(zip(bars, importances[indices])):
        plt.text(imp + 0.002, i, f'{imp:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"\nFeature importance plot saved to 'feature_importance.png'")
    
    plt.close()
    
    return importance_df


######################################################################
# TRAIN AND EVALUATE
######################################################################

def train_and_evaluate(X_train, y_train, X_test, y_test, params, feature_names):
    """
    Train and evaluate Random Forest model
    
    Args:
        X_train, y_train: training data
        X_test, y_test: test data
        params: hyperparameters dict
        feature_names: list of feature names
        
    Returns:
        results: dict of evaluation metrics
        model: trained Random Forest model
    """
    print("\n" + "="*70)
    print("TRAINING AND EVALUATING RANDOM FOREST")
    print("="*70)
    
    # Train model
    rf = RandomForestClassifier(
        **params,
        random_state=42,
        n_jobs=-1
    )
    
    print(f"\nTraining Random Forest with parameters:")
    for param, value in params.items():
        print(f"  {param}: {value}")
    
    rf.fit(X_train, y_train)
    print(f"\nModel trained successfully")
    print(f"Number of trees: {rf.n_estimators}")
    print(f"Number of features: {rf.n_features_in_}")
    
    # Predictions
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)
    
    # Get prediction probabilities
    y_test_proba = rf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\nACCURACY")
    print("="*70)
    print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Test Accuracy:     {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Check for overfitting
    overfit_gap = train_acc - test_acc
    print(f"Overfit gap:       {overfit_gap:.4f} ({overfit_gap*100:.2f}%)")
    if overfit_gap > 0.05:
        print(" Model may be overfitting (gap > 5%)")
    else:
        print(" Model generalization looks good")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nCONFUSION MATRIX")
    print("="*70)
    print(f"\n              Predicted")
    print(f"           No-Diabetes  Diabetes")
    print(f"Actual No     {tn:4d}       {fp:4d}")
    print(f"       Yes    {fn:4d}       {tp:4d}")
    
    # Detailed metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nDETAILED METRICS")
    print("="*70)
    print(f"Precision:    {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:       {recall:.4f} ({recall*100:.2f}%)")
    print(f"Specificity:  {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"F1-Score:     {f1:.4f} ({f1*100:.2f}%)")
    
    # Classification report
    print(f"\nCLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_test, y_test_pred, 
                                target_names=['No Diabetes', 'Diabetes']))
    
    # Feature importance
    importance_df = analyze_feature_importance(rf, feature_names)
    
    # Check if target met
    print("\n" + "="*70)
    if test_acc >= 0.85:
        print(" TARGET ACHIEVED: Accuracy >= 85%")
    else:
        gap = 0.85 - test_acc
        print(f"Target: 85% | Current: {test_acc*100:.2f}%")
        # print(f"Gap: {gap*100:.2f}% improvement needed")
    print("="*70)
    
    results = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'confusion_matrix': cm,
        'feature_importance': importance_df,
        'overfit_gap': overfit_gap
    }
    
    return results, rf


######################################################################
# CROSS-VALIDATION ANALYSIS
######################################################################

def cross_validation_analysis(X, y, params, cv=5):
    """
    Perform detailed cross-validation analysis
    
    Args:
        X: features
        y: labels
        params: model parameters
        cv: number of folds
        
    Returns:
        cv_results: dict with CV metrics
    """
    print("\n" + "="*70)
    print(f"CROSS-VALIDATION ANALYSIS ({cv}-FOLD)")
    print("="*70)
    
    rf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    
    # Perform cross-validation
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
    
    print(f"\nFold-by-fold accuracy:")
    for i, score in enumerate(cv_scores, 1):
        print(f"  Fold {i}: {score:.4f} ({score*100:.2f}%)")
    
    print(f"\n{'='*70}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"Min CV Accuracy:  {cv_scores.min():.4f}")
    print(f"Max CV Accuracy:  {cv_scores.max():.4f}")
    print(f"{'='*70}")
    
    cv_results = {
        'mean': cv_scores.mean(),
        'std': cv_scores.std(),
        'min': cv_scores.min(),
        'max': cv_scores.max(),
        'scores': cv_scores
    }
    
    return cv_results


######################################################################
# MAIN FUNCTION
######################################################################

def main():
    """Complete Random Forest pipeline"""
    print("\n" + "="*70)
    print("RANDOM FOREST DIABETES PREDICTION")
    print("="*70)
    
    # Step 1: Load and preprocess
    X, y, scaler, feature_names = load_and_preprocess_data('datasets/diabetes.csv')
    
    # Step 2: Train/test split
    print("\n" + "="*70)
    print("TRAIN/TEST SPLIT")
    print("="*70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining samples:   {len(X_train)}")
    print(f"Test samples:       {len(X_test)}")
    print(f"Features:           {len(feature_names)}")
    print(f"Train positive %:   {np.mean(y_train)*100:.1f}%")
    print(f"Test positive %:    {np.mean(y_test)*100:.1f}%")
    
    # Step 3: Hyperparameter tuning
    best_params, grid_search = tune_hyperparameters(X_train, y_train, cv=5)
    
    # Step 4: Cross-validation analysis
    cv_results = cross_validation_analysis(X_train, y_train, best_params, cv=5)
    
    # Step 5: Train and evaluate final model
    results, model = train_and_evaluate(X_train, y_train, X_test, y_test, 
                                       best_params, feature_names)
    
    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    
    print(f"\nFINAL SUMMARY:")
    print(f"  Dataset: {len(X)} samples (after preprocessing)")
    print(f"  Features: {feature_names}")
    print(f"  Best Parameters: {best_params}")
    print(f"\nPERFORMANCE:")
    print(f"  CV Accuracy:     {cv_results['mean']:.4f} (±{cv_results['std']:.4f})")
    print(f"  Test Accuracy:   {results['test_accuracy']:.4f}")
    print(f"  Precision:       {results['precision']:.4f}")
    print(f"  Recall:          {results['recall']:.4f}")
    print(f"  F1-Score:        {results['f1_score']:.4f}")
    print(f"\nTOP 3 IMPORTANT FEATURES:")
    for i in range(min(3, len(results['feature_importance']))):
        row = results['feature_importance'].iloc[i]
        print(f"  {i+1}. {row['Feature']} ({row['Percentage']:.2f}%)")
    
    print("\n" + "="*70)
    
    return results, model, grid_search


if __name__ == "__main__":
    results, model, grid_search = main()