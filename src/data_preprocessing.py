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
    to_standardize = [
        'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]

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
        X: d x n array of preprocessed features (8 x n)
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
    
    # Step 2: Drop samples with SkinThickness==0 or Insulin==0
    # Indices: 0 Pregnancies, 1 Glucose, 2 BloodPressure, 3 SkinThickness,
    #          4 Insulin, 5 BMI, 6 DiabetesPedigreeFunction, 7 Age
    print("\n2. Dropping rows where SkinThickness==0 OR Insulin==0 ...")
    valid_mask = (all_features[3, :] != 0) & (all_features[4, :] != 0)
    dropped = all_features.shape[1] - np.sum(valid_mask)
    if dropped > 0:
        print(f"  Dropped {dropped} samples")
    all_features = all_features[:, valid_mask]
    labels = labels[:, valid_mask]

    # Step 3: Select ALL 8 features (including SkinThickness & Insulin)
    print("\n3. Selecting ALL 8 features (including SkinThickness & Insulin)...")
    selected_indices = [0, 1, 2, 3, 4, 5, 6, 7]
    feature_names = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    X = select_features(all_features, selected_indices)
    print(f"  Selected features: {feature_names}")
    print(f"  Shape: {X.shape[0]} features x {X.shape[1]} samples")

    # Step 4: Handle missing values
    print("\n4. Handling missing values...")
    X = handle_missing_values(X, feature_names)
    
    # Step 5: Standardize features
    print("\n5. Standardizing features...")
    X, means, stds = standardize_features(X, feature_names)
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    
    return X, labels, feature_names, means, stds


# Test the preprocessing
if __name__ == "__main__":
    X, y, feature_names, means, stds = preprocess_diabetes_data('datasets/diabetes.csv')
    print(f"\nFinal preprocessed data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Features: {feature_names}")