"""
Data Preprocessing for Diabetes Dataset
"""

import numpy as np
import csv


def load_diabetes_data(filepath='../datasets/diabetes.csv'):
    """
    Load diabetes dataset from CSV file
    
    Returns:
        data: d x n numpy array (features x samples)
        labels: 1 x n numpy array (outcomes)
    """
    data = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append([float(val) for val in row])
    
    data = np.array(data)
    
    print(f"Loaded {data.shape[0]} samples with {data.shape[1]-1} features")
    
    # Separate features and labels
    features = data[:, :-1].T  # d x n
    labels = data[:, -1:].T     # 1 x n
    
    return features, labels


def select_features(features, feature_indices):
    """
    Select subset of features
    
    Args:
        features: d x n array (all features)
        feature_indices: list of indices to keep
        
    Returns:
        selected_features: k x n array (selected features only)
    """
    return features[feature_indices, :]


def handle_missing_values(features, feature_names):
    """
    Handle missing values (represented as 0) in medical features
    Replace with median of non-zero values
    
    Args:
        features: d x n array
        feature_names: list of feature names corresponding to rows
        
    Returns:
        features: d x n array with missing values imputed
    """
    features = features.copy()
    
    # Features that can't actually be 0 (medical impossibility)
    cannot_be_zero = ['Glucose', 'BloodPressure', 'BMI']
    
    for i, name in enumerate(feature_names):
        if name in cannot_be_zero:
            # Find non-zero values
            non_zero_mask = features[i, :] != 0
            if np.sum(~non_zero_mask) > 0:  # If there are zeros
                median_val = np.median(features[i, non_zero_mask])
                features[i, ~non_zero_mask] = median_val
                print(f"  Imputed {np.sum(~non_zero_mask)} missing values in {name} with median={median_val:.2f}")
    
    return features


def standardize_features(features, feature_names):
    """
    Standardize features to mean=0, std=1
    
    Args:
        features: d x n array
        feature_names: list of feature names
        
    Returns:
        standardized_features: d x n array
        means: d x 1 array of means
        stds: d x 1 array of standard deviations
    """
    # Features to standardize (not count variables like Pregnancies)
    to_standardize = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 'BloodPressure']
    
    standardized = features.copy()
    means = np.mean(features, axis=1, keepdims=True)  # d x 1
    stds = np.std(features, axis=1, keepdims=True)    # d x 1
    
    for i, name in enumerate(feature_names):
        if name in to_standardize:
            standardized[i, :] = (features[i, :] - means[i]) / stds[i]
            print(f"  Standardized {name}: mean={means[i,0]:.2f}, std={stds[i,0]:.2f}")
    
    return standardized, means, stds


def preprocess_diabetes_data(filepath='datasets/diabetes.csv'):
    """
    Complete preprocessing pipeline
    
    Returns:
        X: d x n array of preprocessed features (6 x n)
        y: 1 x n array of labels
        feature_names: list of selected feature names
        means: means for standardization
        stds: stds for standardization
    """
    print("="*70)
    print("PREPROCESSING DIABETES DATA")
    print("="*70)
    
    # Step 1: Load data
    print("\n1. Loading data...")
    all_features, labels = load_diabetes_data(filepath)
    
    # Step 2: Select 6 features (drop Insulin and SkinThickness)
    print("\n2. Selecting 6 features (dropping Insulin and SkinThickness)...")
    # Original order: Pregnancies(0), Glucose(1), BloodPressure(2), SkinThickness(3), 
    #                 Insulin(4), BMI(5), DiabetesPedigreeFunction(6), Age(7)
    # Keep: 0, 1, 2, 5, 6, 7
    selected_indices = [0, 1, 2, 5, 6, 7]
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    X = select_features(all_features, selected_indices)
    print(f"  Selected features: {feature_names}")
    print(f"  Shape: {X.shape[0]} features x {X.shape[1]} samples")
    
    # Step 3: Handle missing values
    print("\n3. Handling missing values...")
    X = handle_missing_values(X, feature_names)
    
    # Step 4: Standardize features
    print("\n4. Standardizing features...")
    X, means, stds = standardize_features(X, feature_names)
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    
    return X, labels, feature_names, means, stds


# Test the preprocessing
if __name__ == "__main__":
    X, y, feature_names, means, stds = preprocess_diabetes_data('diabetes.csv')
    print(f"\nFinal preprocessed data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Features: {feature_names}")