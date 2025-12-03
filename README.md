## Multi-Modal Early Disease Detection System

### Motivation
Early detection of chronic diseases like diabetes, heart disease, and stroke is crucial for saving lives and reducing healthcare costs. Current diagnostic methods often require expensive tests and may miss early warning signs. Our system will create an AI-powered early detection platform that combines multiple accessible health indicators to predict disease risk.

With earlier detection and intervention, people can empower themselves to take charge of their own health before things deteriorate. Knowledge is powerful when it comes to your health.

### System Overview
Our Multi-Modal Early Disease Detection System will have the following:
**Input**: Patient demographics, lifestyle factors, basic health measurements, and historical health data
**Output**: Risk scores for diabetes with confidence intervals (heart disease and stroke detection will be added later)

**AI Techniques**: Ensemble learning (Random Forest + XGBoost + Neural Networks) + Explainable AI using medLlamma2 (https://ollama.com/library/medllama2) and Meditron (https://ollama.com/library/meditron)
Interface: Web-based dashboard

#### Innovation & Impact
**Real-World Application**: Could be deployed in clinics, especially in underserved areas with limited diagnostic equipment
**Humanitarian Impact**: Early intervention could save thousands of lives annually and reduce healthcare disparities
**Academic Impact**: More people contributing to datasets means better models for the future

#### How to run
**How-to-guide**: Contains the step by step details on how to run this project locally
**TLDR**: 
   - Run the `run_pipeline.py` file inside the `pipeline` folder
   - Start the web server by running `streamlit run .\app.py` in the `web_app` folder 

### AI Techniques (Multiple Integrated Methods)
**Core ML Pipeline**
1. Ensemble Learning
   - Random Forest: Handles feature aggregation and provides feature importance
   - XGBoost: Gradient boosting for complex pattern recognition in features
   - Neural Networks: Deep feature learning and non-linear relationships
2. Explainable AI
   - MedLlamma2:  Instance-specific explanations
   - Meditron:  Medical question answering / Disease information query

### Datasets & Tools
#### External Tools/libraries:  
We’ll be using the Transformers library to deploy the local models for medical explainability and sourcing open source models from Hugging face to give better medical feedback. To deploy models locally, we also need several other python libraries running Transformers, such as: PyTorch, TensorFlow and Flax.  

#### Tools & Libraries
- ML Frameworks: scikit-learn, XGBoost, TensorFlow/Keras, Transformers
- Data Processing: pandas, numpy, scikit-learn preprocessing
- Visualization: matplotlib, seaborn, streamlit
- Web Framework: Streamlit or Flask for deployment

#### Primary Datasets
| Dataset | Source | Size | Target Disease | Key Features |
| Diabetes | UCI | 768 samples | Diabetes | Glucose, BMI, Age, Pregnancies |
| Heart Disease Cleveland | UCI | 303 samples | Heart Disease | Chest pain, cholesterol, ECG results |
| Stroke Prediction Dataset | Kaggle | 5,110 samples | Stroke | Hypertension, heart disease, BMI | 

#### Secondary Datasets (Backup and cross validation):
**Diabetes:**
- https://archive.ics.uci.edu/dataset/34/diabetes 
- https://figshare.com/articles/dataset/diabetes_csv/25421347?file=45084472
- https://data.mendeley.com/datasets/m8cgwxs9s6/2
- https://data.mendeley.com/datasets/wj9rwkp9c2/1
**Heart Disease:**
- https://ieee-dataport.org/documents/cardiovascular-disease-dataset
**Stroke:**
- https://data.mendeley.com/datasets/s2nh6fm925/1

(We’re proposing more secondary diabetes datasets for now as our MVP will focus on diabetes, but we’ll source more datasets for Heart Disease and Stroke as needed)

### Project Timeline

#### Milestone 1 (October 19, 2025)
Goal: Complete data preprocessing and baseline models
Deliverables:
- All datasets downloaded, cleaned, and integrated
- Initial Exploratory Data Analysis with quick insights
- Feature engineering pipeline initiation
- Baseline models trained (individual algorithms)
- Initial evaluation metrics calculated

#### Milestone 2 (November 16, 2025)
Goal: Complete ensemble system with explainability
Deliverables:
- Ensemble implemented and optimized
- Medical explainability integration
- Cross-validation results with confidence intervals
- Web application prototype with real-time predictions
- Model performance comparison study
 

### Minimal Viable System
Core Functionality
1. Single Disease Prediction: Start with diabetes prediction using diabetes dataset
2. Basic Ensemble: Random Forest + XGBoost combination
3. Simple Explanations: Feature importance rankings
4. Command-line Interface: Input patient data, get risk score
 
#### Success Criteria
- Accuracy: >85% on test set with proper cross-validation
- Explainability: Clear identification of top 5 risk factors
- Usability: input patient data and get results in a reasonably quick time
- Reproducibility: Complete code with documentation and environment setup
 
#### Expansion Path
1. Add heart disease and stroke prediction - Done
2. Implement advanced explainability - In Progress
3. Build web interface - Done
4. Add certainty quantification - Done
5. Integrate multiple data sources - Done
 
 

