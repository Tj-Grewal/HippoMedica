#!/usr/bin/env python3
"""
Complete ML Pipeline Orchestrator for Multi-Modal Disease Detection
Executes all pipeline steps for diabetes, heart disease, and stroke detection.

Pipeline: Download → EDA → Preprocessing → Training → Storage

Usage:
    python run_pipeline.py                    # Run full pipeline for all diseases
    python run_pipeline.py --skip-download    # Skip download step (use existing data)
    python run_pipeline.py --skip-eda         # Skip exploratory analysis
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import pipeline steps
import importlib.util
import sys
from pathlib import Path

# Dynamic imports to handle numbered filenames
def import_module(module_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Get pipeline directory
pipeline_dir = Path(__file__).parent

# Import each step module
step_01 = import_module(pipeline_dir / "01_data_download.py", "step_01")
step_02 = import_module(pipeline_dir / "02_exploratory_analysis.py", "step_02")
step_03 = import_module(pipeline_dir / "03_preprocessing.py", "step_03")
step_04 = import_module(pipeline_dir / "04_model_training.py", "step_04")
step_05 = import_module(pipeline_dir / "05_model_storage.py", "step_05")

# Extract classes
DataDownloader = step_01.DataDownloader
ExploratoryAnalysis = step_02.ExploratoryAnalysis
DataPreprocessor = step_03.DataPreprocessor
ModelTrainer = step_04.ModelTrainer
ModelStorage = step_05.ModelStorage

class PipelineOrchestrator:
    """Orchestrates the complete ML pipeline."""
    
    def __init__(self, skip_download=False, skip_eda=False):
        self.skip_download = skip_download
        self.skip_eda = skip_eda
        self.start_time = None
        self.results = {}
    
    def print_header(self):
        """Print pipeline header."""
        print("\n" + "="*80)
        print(" "*15 + "MULTI-MODAL DISEASE DETECTION - ML PIPELINE")
        print("="*80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Diseases: Diabetes, Heart Disease, Stroke")
        print("="*80 + "\n")
    
    def print_footer(self):
        """Print pipeline footer with summary."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        print("\n" + "="*80)
        print(" "*25 + "PIPELINE COMPLETE!")
        print("="*80)
        print(f"Total Time: {minutes}m {seconds}s")
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if 'best_models' in self.results and self.results['best_models']:
            print("\n[!] Best Models by Disease:")
            for disease_key in ['diabetes', 'heart_disease', 'stroke']:
                if disease_key in self.results['best_models']:
                    info = self.results['best_models'][disease_key]
                    disease_display = disease_key.replace('_', ' ').title()
                    print(f"   {disease_display}: {info['name']}")
                    print(f"      Accuracy: {info['accuracy']:.4f} ({info['accuracy']*100:.2f}%)")
        
        if 'storage_complete' in self.results:
            print(f"\n[*] Models saved to: models/trained_models/")
            saved_diseases = []
            if 'all_results' in self.results:
                for key in ['diabetes', 'heart_disease', 'stroke']:
                    if key in self.results['all_results'] and self.results['all_results'][key] is not None:
                        saved_diseases.append(f"{key}_*.pkl")
            if saved_diseases:
                for disease_file in saved_diseases:
                    print(f"   - {disease_file}")
        
        print("\n[>] Next Steps:")
        print("   1. Review visualizations in: pipeline/outputs/visualizations/")
        print("   2. Check model performance in: models/metadata/")
        print("   3. Run web app: cd web_app && streamlit run app.py")
        print("="*80 + "\n")
    
    def step_1_download(self):
        """Step 1: Download datasets for all diseases."""
        if self.skip_download:
            print(">  Skipping Step 1: Data Download (using existing data)\n")
            return True
        
        try:
            downloader = DataDownloader()
            success = downloader.download_all()
            
            if success:
                self.results['download_complete'] = True
                return True
            else:
                print("[X] Step 1 failed: Could not download all datasets")
                return False
                
        except Exception as e:
            print(f"[X] Error in Step 1: {e}")
            return False
    
    def step_2_eda(self):
        """Step 2: Exploratory Data Analysis for all diseases."""
        if self.skip_eda:
            print(">  Skipping Step 2: Exploratory Analysis\n")
            return True
        
        try:
            analyzer = ExploratoryAnalysis()
            success = analyzer.analyze_all()
            
            if success:
                self.results['eda_complete'] = True
                return True
            else:
                print("[X] Step 2 failed: EDA error")
                return False
                
        except Exception as e:
            print(f"[X] Error in Step 2: {e}")
            return False
    
    def step_3_preprocessing(self):
        """Step 3: Data Preprocessing for all diseases."""
        try:
            preprocessor = DataPreprocessor()
            success = preprocessor.preprocess_all()
            
            if success:
                self.results['preprocessing_complete'] = True
                return True
            else:
                print("[X] Step 3 failed: Preprocessing error")
                return False
                
        except Exception as e:
            print(f"[X] Error in Step 3: {e}")
            return False
    
    def step_4_training(self):
        """Step 4: Model Training for all diseases."""
        try:
            trainer = ModelTrainer()
            all_results = trainer.train_all()
            
            if all_results:
                self.results['all_results'] = all_results
                
                # Calculate summary stats for best models (only for successful trainings)
                best_models = {}
                for disease_key in ['diabetes', 'heart_disease', 'stroke']:
                    if disease_key in all_results and all_results[disease_key] is not None:
                        model_results = all_results[disease_key]['results']
                        best_model_name = max(model_results.keys(), 
                                             key=lambda x: model_results[x]['accuracy'])
                        best_models[disease_key] = {
                            'name': best_model_name,
                            'accuracy': model_results[best_model_name]['accuracy']
                        }
                
                self.results['best_models'] = best_models
                
                # Check if at least one disease was successfully trained
                successful_count = sum(1 for v in all_results.values() if v is not None)
                if successful_count > 0:
                    return True
                else:
                    print("[X] Step 4 failed: No models trained successfully")
                    return False
            else:
                print("[X] Step 4 failed: Training error")
                return False
                
        except Exception as e:
            print(f"[X] Error in Step 4: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step_5_storage(self):
        """Step 5: Model Storage for all diseases."""
        try:
            storage = ModelStorage()
            success = storage.save_all(self.results['all_results'])
            
            if success:
                self.results['storage_complete'] = True
                return True
            else:
                print("[X] Step 5 failed: Storage error")
                return False
                
        except Exception as e:
            print(f"[X] Error in Step 5: {e}")
            return False
    
    def run(self):
        """Execute the complete pipeline."""
        self.start_time = datetime.now()
        self.print_header()
        
        # Execute pipeline steps
        steps = [
            ("Step 1: Data Download", self.step_1_download),
            ("Step 2: Exploratory Analysis", self.step_2_eda),
            ("Step 3: Data Preprocessing", self.step_3_preprocessing),
            ("Step 4: Model Training", self.step_4_training),
            ("Step 5: Model Storage", self.step_5_storage)
        ]
        
        for step_name, step_func in steps:
            try:
                success = step_func()
                if not success:
                    print(f"\n[X] Pipeline stopped at: {step_name}")
                    print("   Please fix the errors and try again.\n")
                    return False
            except Exception as e:
                print(f"\n[X] Unexpected error in {step_name}: {e}")
                return False
        
        # Print summary
        self.print_footer()
        return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run the complete multi-modal disease detection ML pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                    # Run full pipeline for all diseases
  python run_pipeline.py --skip-download    # Skip download (use existing data)
  python run_pipeline.py --skip-eda         # Skip exploratory analysis
        """
    )
    
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip data download step (use existing data)')
    parser.add_argument('--skip-eda', action='store_true',
                       help='Skip exploratory data analysis step')
    
    args = parser.parse_args()
    
    # Run pipeline
    orchestrator = PipelineOrchestrator(
        skip_download=args.skip_download,
        skip_eda=args.skip_eda
    )
    
    success = orchestrator.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
