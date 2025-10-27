"""
KNN Baseline Model for Diabetes Prediction
Using Scikit-Learn (Simplified Output)
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


######################################################################
# PREPROCESSING
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
# KNN CROSS-VALIDATION
######################################################################

def find_optimal_k(X_train, y_train, k_range=range(1, 51), cv=5):
    """
    Find optimal K using cross-validation
    
    Args:
        X_train: training features
        y_train: training labels
        k_range: range of k values to test
        cv: number of CV folds
        
    Returns:
        best_k: optimal k value
        cv_scores: dict of k -> mean CV accuracy
    """
    print("\n" + "="*70)
    print("FINDING OPTIMAL K USING CROSS-VALIDATION")
    print("="*70)
    
    print(f"\nTesting k from {k_range.start} to {k_range.stop-1}")
    print(f"Using {cv}-fold cross-validation\n")
    
    cv_scores = {}
    best_score = 0
    best_k = 1
    
    print("K   | CV Accuracy | Std Dev")
    print("----|-------------|----------")
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='accuracy')
        
        mean_score = scores.mean()
        std_score = scores.std()
        cv_scores[k] = mean_score
        
        if mean_score > best_score:
            best_score = mean_score
            best_k = k
        
        # Print every 5th k
        if k == 1 or k % 5 == 0:
            print(f"{k:3d} | {mean_score:.4f}      | {std_score:.4f}")
    
    print(f"\n{'='*70}")
    print(f"Optimal K: {best_k}")
    print(f"Best CV Accuracy: {cv_scores[best_k]:.4f} ({cv_scores[best_k]*100:.2f}%)")
    print(f"{'='*70}")
    
    return best_k, cv_scores


######################################################################
# TRAIN AND EVALUATE
######################################################################

def train_and_evaluate(X_train, y_train, X_test, y_test, k):
    """
    Train and evaluate KNN model
    
    Args:
        X_train, y_train: training data
        X_test, y_test: test data
        k: number of neighbors
        
    Returns:
        results: dict of evaluation metrics
        model: trained KNN model
    """
    print("\n" + "="*70)
    print(f"TRAINING AND EVALUATING KNN (k={k})")
    print("="*70)
    
    # Train model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    print(f"\nModel trained with k={k}")
    
    # Predictions
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\nACCURACY")
    print("="*70)
    print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Test Accuracy:     {test_acc:.4f} ({test_acc*100:.2f}%)")
    
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
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nDETAILED METRICS")
    print("="*70)
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Classification report
    print(f"\nCLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_test, y_test_pred, 
                                target_names=['No Diabetes', 'Diabetes']))
    
    # Check if target met
    print("="*70)
    if test_acc >= 0.85:
        print("TARGET ACHIEVED: Accuracy >= 85%")
    else:
        gap = 0.85 - test_acc
        print(f"Target: 85% | Current: {test_acc*100:.2f}%")
        print(f"Gap: {gap*100:.2f}% improvement needed")
    print("="*70)
    
    results = {
        'k': k,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }
    
    return results, knn


######################################################################
# MAIN FUNCTION
######################################################################

def main():
    """Complete KNN pipeline"""
    print("\n" + "="*70)
    print("KNN DIABETES PREDICTION (SCIKIT-LEARN)")
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
    
    # Step 3: Find optimal k
    best_k, cv_scores = find_optimal_k(X_train, y_train, 
                                       k_range=range(1, 51), cv=5)
    
    # Step 4: Train and evaluate
    results, model = train_and_evaluate(X_train, y_train, X_test, y_test, best_k)
    
    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    
    print(f"\nFINAL SUMMARY:")
    print(f"  Dataset: 768 samples")
    print(f"  Features: {feature_names}")
    print(f"  Optimal K: {best_k}")
    print(f"  Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1-Score: {results['f1_score']:.4f}")
    
    return results, model


if __name__ == "__main__":
    results, model = main()