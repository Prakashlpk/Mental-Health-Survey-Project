Mental Health Survey Project
Deep Learning model to predict depression from survey data
Project Documentation
1. Introduction
Depression is a growing public health challenge. Early detection can help in timely intervention and treatment. This project aims to predict whether an individual may experience depression based on demographic, academic, lifestyle, and medical history factors. We use a deep learning model to identify hidden patterns in survey data and deploy it via a Streamlit web app for real-time predictions hosted on AWS.
2. Problem Statement
The objective is to build a predictive model that classifies individuals into high-risk or low-risk categories for depression. The model must ensure fairness across demographics (gender, age, profession, city), achieve high predictive accuracy using balanced datasets, and be practical for healthcare, corporate, and governmental decision-making.
3. Dataset
•	Source: Mental Health Survey Data (CSV format)
•	Features: 
o	Demographics: Age, Gender, City
o	Academic & Work: Degree, Profession, Work/Study status
o	Pressures: Academic, Work, Financial Stress
o	Lifestyle: Dietary habits, Sleep duration, Work/Study hours
o	History: Family history of mental illness, Suicidal thoughts
•	Target Variable: Depression (0 = No, 1 = Yes)
4. Data Preprocessing
The following preprocessing steps were applied on training & test datasets:
•	Cleaning & Standardization:
o	Corrected invalid entries in categorical fields (City, Profession, Degree, Sleep Duration, Dietary Habits).
o	Removed irrelevant columns like Name and ID.
•	Handling Missing Values:
o	Median imputation for numeric fields.
o	Mode imputation for categorical fields.
•	Encoding & Transformation:
o	One-hot encoding for categorical variables.
o	Log transformation for skewed features (Academic Pressure, CGPA, Study Satisfaction).
•	Balancing:
o	Applied SMOTE stratified by Age Group × Profession to handle class imbalance.
•	Scaling:
o	StandardScaler used for numerical feature scaling.
5. Model Development
•	Framework: TensorFlow (Keras API)
•	Architecture:
o	Input layer: 100+ encoded features
o	Hidden Layers: Dense layers (256 → 128 → 64 → 32) with ReLU activation, Batch Normalization, Dropout (0.15–0.3)
o	Output Layer: 1 neuron with sigmoid activation (binary classification)
•	Training:
o	Loss: Binary Crossentropy
o	Optimizer: Adam (lr=0.001)
o	Metrics: Accuracy, Precision, Recall, AUC
o	Early Stopping: patience = 10
6. Model Evaluation
6.1 Overall Performance Metrics
Validation Metrics (23,999 samples):
•	Accuracy: 97%
•	Precision: 0.97 (Class 0), 0.97 (Class 1)
•	Recall: 0.97 (Class 0), 0.96 (Class 1)
•	F1-score: 0.97 (Class 0), 0.96 (Class 1)
•	AUC: 0.9946
•	Confusion Matrix: [[11,598, 402], [437, 11,562]]
Test Metrics (48,728 samples):
•	Accuracy: 96%
•	Precision: 0.96 (Class 0), 0.96 (Class 1)
•	Recall: 0.96 (Class 0), 0.96 (Class 1)
•	F1-score: 0.96 (Class 0), 0.96 (Class 1)
•	AUC: 0.9940
•	Confusion Matrix: [[23,452, 912], [973, 23,391]]
6.2 Demographic Fairness Analysis
Gender-based Performance:
•	Female (24,708 samples): Accuracy 97%, Precision 0.96-0.97, Recall 0.96-0.97
•	Male (28,913 samples): Accuracy 96%, Precision 0.96-0.97, Recall 0.96-0.97
Age Group Performance:
•	<18 (839 samples): Accuracy 86%, F1-score 0.77-0.90
•	18-29 (8,325 samples): Accuracy 86%, F1-score 0.86-0.87
•	30-44 (17,170 samples): Accuracy 97%, F1-score 0.97
•	45-59 (22,394 samples): Accuracy 100%, F1-score 1.00
6.3 Professional Category Analysis
Student Population (7,047 samples):
•	Accuracy: 87%
•	Precision: 0.90 (Class 0), 0.84 (Class 1)
•	F1-score: 0.86-0.87
Working Professionals (41,681 samples):
•	Accuracy: 98%
•	Precision: 0.97 (Class 0), 0.99 (Class 1)
•	F1-score: 0.98
6.4 Profession-Specific Performance
High-performing professions (>98% accuracy):
•	Digital Marketer: 100% accuracy, perfect precision/recall
•	Teacher (11,914 samples): 99% accuracy, F1-score 0.99
•	Pharmacist: 99% accuracy
•	HR Manager: 99% accuracy
•	Chef, Electrician, Manager: 99% accuracy each
Moderate-performing professions:
•	Student: 87% accuracy (largest challenging group)
•	Software Engineer: 97% accuracy
•	Content Writer: 99% accuracy
6.5 Educational Background Analysis
Performance across degree types shows consistent high accuracy (95-100%):
•	PhD, ME, MED: 98-99% accuracy
•	MBBS, MD: 97% accuracy
•	Engineering degrees (BE, BTECH): 97% accuracy
•	Business degrees (BBA, MBA): 97% accuracy
•	CLASS 12 (7,143 samples): 95% accuracy
6.6 Geographic Distribution Analysis
City-wise performance demonstrates model robustness across urban centers:
•	Major metros: 97-98% accuracy (Mumbai, Delhi, Bangalore, Chennai)
•	Tier-2 cities: 96-98% accuracy (Pune, Ahmedabad, Hyderabad)
•	Smaller cities: 96-98% accuracy maintained
7. Deployment
The project has been successfully deployed with the following options:
•	Streamlit Application:
o	Mental Health Assessment Page: Users can input personal, academic, lifestyle, and history details to get risk predictions.
o	Business Use Cases Page: Modules for Healthcare Providers, Mental Health Clinics, Corporate Wellness Programs, and Government/NGOs.
•	Deployment Environment:
o	The Streamlit app is deployed on AWS (Elastic BeanstalEC2) for real-time global access.
o	Ensures scalability, reliability, and security for end users.
8. Results & Insights
8.1 Model Performance Highlights
•	Exceptional Overall Performance: 96-97% accuracy on large-scale test data
•	Demographic Fairness: Consistent performance across gender and professional categories
•	Age-related Patterns: Higher accuracy in middle-aged groups (30-59 years)
•	Professional Insights: Students show more prediction complexity, requiring targeted interventions
8.2 Clinical and Practical Applications
•	Healthcare Providers: Enables early screening of high-risk patients with 96% accuracy
•	Mental Health Clinics: Prioritization of treatment queues with demographic-aware predictions
•	Corporate Wellness: Employee mental health monitoring across professional categories
•	Educational Institutions: Targeted student mental health support based on risk assessment
•	Government & NGOs: Population-level risk analytics with geographic insights
8.3 Model Reliability
•	Consistent Cross-validation: Validation and test metrics align closely (97% vs 96%)
•	Balanced Performance: Equal precision/recall for both depression classes
•	Demographic Robustness: Stable performance across 30+ professions and 25+ cities
9. Deliverables
•	Source Code: Data preprocessing, model training, and deployment scripts
•	Trained Model: mental_health_survey_final.keras, scaler, and feature columns
•	Streamlit App: Interactive web application with demographic-aware predictions
•	Comprehensive Evaluation: Detailed metrics across demographics, professions, and locations
•	Documentation: Complete analysis of model performance and fairness assessment
•	AWS Deployment: Hosted app accessible for real-time predictions with enterprise-grade reliability
10. Future Enhancements
•	Longitudinal Analysis: Track prediction accuracy over time
•	Regional Customization: City-specific model fine-tuning
•	Professional Risk Profiles: Occupation-specific intervention strategies
•	Mobile Integration: Smartphone app for broader accessibility

