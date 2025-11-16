"""
Neural network Model for Diabetes Prediction using scikit-learn
Use "--retrain" flag to force the retraining of the model
"""

"""
- train_test_split: split datasets into training and test
- GridSearchCV: find the combination of parameters that produce best cross-validated score (return optimized model)
	https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
- StratifiedKFold: cross-validation that preserve class proportions in each fold. Each fold is representative of the overall distribution
- MLPCLassifier: feed-forward neural network implementation
- os: functions to help save the model
- pickle: save model to avoid retraining
"""
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import pickle

# Reusing preprocessing from knn_baseline
from knn_baseline import load_and_preprocess_data

##########################################
# Helpers function to save and load trained models
##########################################
def save_model(save_path, model, scaler, feature_names, params=None):
	os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
	with open(save_path, 'wb') as f:
		pickle.dump({
			'model': model,
			'scaler': scaler,
			'feature_names': feature_names,
			'params': params if params is not None else (model.get_params() if hasattr(model, 'get_params') else None)
		}, f)
	print(f"Saved model to {save_path}")

# Return (model, scaler, feature_names) or (None, None, None) if not found.
def load_saved_model(save_path):
	if os.path.exists(save_path):
		with open(save_path, 'rb') as f:
			obj = pickle.load(f)
		model = obj.get('model')
		scaler = obj.get('scaler')
		feature_names = obj.get('feature_names')
		params = obj.get('params')
		print(f"Loaded saved model from {save_path}")
		return model, scaler, feature_names, params
	return None, None, None, None

##########################################
# Evaluation helper
##########################################
def evaluate_model(model, X_test, y_test):
	y_test_pred = model.predict(X_test)
	test_acc = accuracy_score(y_test, y_test_pred)
	cm = confusion_matrix(y_test, y_test_pred)
	print("\n" + "="*70)
	print("EVALUATION")
	print("="*70)
	print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
	print("\nConfusion matrix:\n", cm)
	print("\nClassification report:\n", classification_report(y_test, y_test_pred, target_names=['No Diabetes','Diabetes']))
	return {'test_accuracy': test_acc, 'confusion_matrix': cm}


##########################################
# Multi Layer Perception Cross-validation
##########################################

def find_best_mlp(X_train, y_train, param_grid=None, cv=5):
	"""
	Grid search for best MLP hyperparameters.

	Args:
		X_train: training features
		y_train: training labels
		cv: number of folds

	Returns:
		best_model
		best_params
	"""
	print("\n" + "="*70)
	print("FINDING BEST MLP VIA GRIDSEARCH")
	print("="*70)

	param_grid = {
		'hidden_layer_sizes': [(32,), (64,), (128,), (32, 32), (64, 64), (128, 128)],
		'activation': ['logistic', 'relu'],
		'alpha': [1e-3, 1e-4, 1e-5],		# L2 regulation, penalize large weights ("lambda" in the slides)
		'learning_rate_init': [1e-2, 1e-3, 1e-4],
	}

	base = MLPClassifier(max_iter=2000, early_stopping=True, random_state=42)
	skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
	gs = GridSearchCV(base, param_grid, scoring='accuracy', cv=skf, n_jobs=-1, verbose=1)
	gs.fit(X_train, y_train)

	print(f"Best params: {gs.best_params_}")
	print(f"Best CV score: {gs.best_score_:.4f}")
	return gs.best_estimator_, gs.best_params_

##########################################
# Train and evaluate
##########################################
def train_and_evaluate(X_train, y_train, X_test, y_test, model):
	"""
	Train the MLP model and print evaluation metrics.
	
	Args:
		X_train: training features
		y_train: training labels
		X_test: final feature matrix used for testing
		y_test: true labels for test sets
		model: chosen hyperparameters, learned weights and biases

	Returns:
		results dict
		trained model
		cv results: mean validation score across the cv folds for the best combination
	"""
	print("\n" + "="*70)
	print("TRAINING AND EVALUATING MLP")
	print("="*70)

	model.fit(X_train, y_train)
	print("\nModel training complete")

	y_train_pred = model.predict(X_train)
	y_test_pred = model.predict(X_test)

	train_acc = accuracy_score(y_train, y_train_pred)
	test_acc = accuracy_score(y_test, y_test_pred)

	# Printing out results
	print(f"\nACCURACY")
	print("="*70)
	print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
	print(f"Test Accuracy:     {test_acc:.4f} ({test_acc*100:.2f}%)")

	cm = confusion_matrix(y_test, y_test_pred)
	tn, fp, fn, tp = cm.ravel()

	print(f"\nCONFUSION MATRIX")
	print("="*70)
	print(f"\n              Predicted")
	print(f"           No-Diabetes  Diabetes")
	print(f"Actual No     {tn:4d}       {fp:4d}")
	print(f"       Yes    {fn:4d}       {tp:4d}")

	precision = tp / (tp + fp) if (tp + fp) > 0 else 0
	recall = tp / (tp + fn) if (tp + fn) > 0 else 0
	f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

	print(f"\nDETAILED METRICS")
	print("="*70)
	print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
	print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
	print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")

	print(f"\nCLASSIFICATION REPORT")
	print("="*70)
	print(classification_report(y_test, y_test_pred, target_names=['No Diabetes', 'Diabetes']))

	results = {
		'train_accuracy': train_acc,
		'test_accuracy': test_acc,
		'precision': precision,
		'recall': recall,
		'f1_score': f1,
		'confusion_matrix': cm
	}

	return results, model

##########################################
# Main function
##########################################

def main(filepath='datasets/diabetes.csv', save_path='models/mlp_diabetes.pkl', force_retrain=False):
	print("\n" + "="*70)
	print("MLP DIABETES PREDICTION")
	print("="*70)

	# Step 1: Load & preprocess
	print("\n1. Loading and preprocessing data")
	X, y, scaler, feature_names = load_and_preprocess_data(filepath)

	# Step 2: Load saved model if present (unless forced to retrain)
	print("\n2. Checking for saved model")
	model, saved_scaler, saved_features, saved_params = load_saved_model(save_path)

	# If a saved model exists and retraining not requested -> show params, evaluate and return
	if model is not None and not force_retrain:
		print("\nSaved model found — skipping training.")
		if saved_params:
			print("\nSaved hyperparameters:")
			print(saved_params)
		print("\n3. Train/test split")
		X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size=0.2, random_state=42, stratify=y
		)
		results = evaluate_model(model, X_test, y_test)
		return results, model

	# Step 3: No saved model -> train new one
	print("\nNo saved model found or retraining requested — running CV, training and saving the model.")
	print("\n3. Train/test split")
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42, stratify=y
	)

	print("\n4. Hyperparameter search (GridSearchCV)")
	best_model, best_params = find_best_mlp(X_train, y_train, cv=5)

	print("\n5. Training final model and evaluating")
	results, trained_model = train_and_evaluate(X_train, y_train, X_test, y_test, best_model)

	print("\n6. Saving trained model")
	save_model(save_path, trained_model, scaler, feature_names, params=best_params)

	print("\n" + "="*70)
	print("PIPELINE COMPLETE")
	print("="*70)
	print(f"\nFinal Test Accuracy: {results.get('test_accuracy')}\n")

	return results, trained_model

# Add option to force retraining of the model
if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Train / load MLP diabetes model")
	parser.add_argument('--retrain', action='store_true', help='Force retrain even if a saved model exists')
	args = parser.parse_args()

	results, model = main(force_retrain=args.retrain)