import requests
import pandas as pd
import numpy as np
from pathlib import Path

# Downloads data using URL's provided
class SimpleDataDownloader:
    
    def __init__(self, output_dir="../datasets/"):
        
        # Initialize directory and make if not exists
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_sources = {
            "diabetes": {
                "name": "Pima Indians Diabetes",
                "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
                "filename": "diabetes.csv",
                "columns": ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
            },
        }
        
    def download_from_url(self, url, filename, columns = None):
        try:
            print(f" - Downloading {filename} from URL...\n")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            filepath = self.output_dir / filename
            
            if url.endswith('.csv'):
                with open(filepath, 'wb') as f:
                    f.write(response.content)
            else:
                # TODO - handle raw data if the format is not csv
                print("Implement raw data handling...\n")
                return 
            
            print(f" - data downloaded successfully: {filename}\n")
            return True
        
        except Exception as e:
            print(f" - Failed to download {filename}: {e}\n")
            return False
                
    def download_datasets(self):
        print("Initializing data download...\n")
        
        results = {}
        
        for dataset_key, config in self.data_sources.items():
            print(f"Processing {config['name']}...\n")
            
            if 'url' in config:
                success = self.download_from_url(
                    config['url'],
                    config['filename'],
                    config.get('columns')
                )
            else:
                success = False
                print(f"Error while downloading {dataset_key}\n")
                
            results[dataset_key] = success
            
        return results
    
    def validate_data(self):
        print("\nRunning data validation on downloaded datasets...\n")
        
        csv_files = list(self.output_dir.glob("*.csv"))
        valid_results = {}
        
        for csv in csv_files:
            try:
                df = pd.read_csv(csv)
                dataset_name = csv.stem
                
                valid_results[dataset_name] = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "size_kb": csv.stat().st_size / (1024),
                    "missing_pct": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                }    
                print(f" - {dataset_name}: {len(df):,} rows, {len(df.columns)} columns")

            except Exception as e:
                print(f"Data validation failed for {csv}: {e}")
                valid_results[csv.stem] = {"error": str(e)}

        return valid_results

    def print_summary(self, datasets, validations):
        report_lines = [
            "# Dataset Download Summary",
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Download Results",
            ""
        ]
        
        for dataset, success in datasets.items():
            status = "Success" if success else "Failed"
            report_lines.append(f"- {dataset}: {status}")
        
        report_lines.extend([
            "",
            "## Dataset Overview",
            "",
            "| Dataset | Rows | Columns | Size (KB) | Missing % |",
            "|---------|------|---------|-----------|-----------|"
        ])
        
        for dataset, stats in validations.items():
            if "error" not in stats:
                rows = stats["rows"]
                cols = stats["columns"]
                size_kb = stats["size_kb"]
                missing_pct = stats["missing_pct"]
                
                report_lines.append(f"| {dataset} | {rows:,} | {cols} | {size_kb:.1f} KB | {missing_pct:.1f}% |")

        print("\n".join(report_lines))

def main():
    
    downloader = SimpleDataDownloader()
    
    datasets = downloader.download_datasets()
    validation = downloader.validate_data()
    downloader.print_summary(datasets, validation)
    
    successful = sum(datasets.values())
    total = len(datasets)
    
    print(f" Successful: {successful}/{total} datasets")
    
    print("Ran main")


if __name__ == "__main__":
    main()