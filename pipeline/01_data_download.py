#!/usr/bin/env python3
"""
Step 1: Data Download
Downloads disease datasets from direct URLs (no Kaggle authentication required).
"""

import pandas as pd
import requests
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataDownloader:
    """Downloads disease datasets from direct URLs."""
    
    def __init__(self, output_dir="../datasets/raw"):
        self.output_dir = Path(__file__).parent / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define all data sources with consistent structure
        self.data_sources = {
            'diabetes': {
                "name": "Pima Indians Diabetes",
                "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
                "filename": "diabetes.csv",
                "columns": ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
                "has_header": False
            },
            'heart_disease': {
                "name": "Heart Disease",
                "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
                "filename": "heart.csv",
                "columns": ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'],
                "has_header": False
            },
            'stroke': {
                "name": "Stroke Prediction Dataset",
                "url": "https://gist.githubusercontent.com/aishwarya8615/d2107f828d3f904839cbcb7eaa85bd04/raw/healthcare-dataset-stroke-data.csv",
                "filename": "healthcare-dataset-stroke-data.csv",
                "has_header": True
            },
        }
    
    def download_dataset(self, disease_key):
        """Download a specific disease dataset.
        
        Args:
            disease_key: One of 'diabetes', 'heart_disease', 'stroke'
        """
        if disease_key not in self.data_sources:
            print(f"[X] Unknown disease: {disease_key}")
            return None
        
        source = self.data_sources[disease_key]
        print(f"\n[>] Downloading {source['name']}...")
        
        filepath = self.output_dir / source['filename']
        
        # Check if file already exists
        if filepath.exists():
            try:
                df = pd.read_csv(filepath)
                print(f"[OK] Already exists: {source['name']}")
                print(f"   File: {filepath}")
                print(f"   Rows: {len(df):,}")
                return filepath
            except:
                print(f"   Existing file corrupt, redownloading...")
        
        # Download from URL
        try:
            response = requests.get(source['url'], timeout=30)
            response.raise_for_status()
            
            if source.get('has_header', False):
                # Dataset already has headers
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                df = pd.read_csv(filepath)
            else:
                # Parse CSV data without headers
                lines = response.text.strip().split('\n')
                data_rows = []
                
                for line in lines:
                    if line.strip():
                        row = [item.strip().strip('"\'') for item in line.split(',')]
                        row = ['' if item == '?' else item for item in row]
                        data_rows.append(row)
                
                # Create DataFrame with proper columns
                df = pd.DataFrame(data_rows, columns=source['columns'])
                df.to_csv(filepath, index=False)
            
            print(f"[OK] Downloaded: {source['name']}")
            print(f"   File: {filepath}")
            print(f"   Rows: {len(df):,}")
            
            return filepath
            
        except Exception as e:
            print(f"[X] Failed to download {source['name']}: {e}")
            return None
    
    def download_all(self):
        """Download all disease datasets."""
        print("="*80)
        print("STEP 1: DATA DOWNLOAD - ALL DATASETS")
        print("="*80)
        print(f"\n[*] Output directory: {self.output_dir.absolute()}")
        
        results = {}
        for disease_key in ['diabetes', 'heart_disease', 'stroke']:
            results[disease_key] = self.download_dataset(disease_key)
        
        # Summary
        successful = sum(1 for v in results.values() if v is not None)
        print(f"\n{'='*80}")
        print(f"[OK] Downloaded {successful}/3 datasets successfully")
        print(f"{'='*80}\n")
        
        return results

def main():
    """Execute Step 1: Data Download"""
    downloader = DataDownloader()
    results = downloader.download_all()
    return results

if __name__ == "__main__":
    main()
