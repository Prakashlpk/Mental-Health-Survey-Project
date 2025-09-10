Mental Health Survey Project
Deep Learning Model to Predict Depression from Survey Data
1. Introduction
Depression is a growing public health challenge. Early detection can help in timely intervention and treatment. This project aims to predict whether an individual may experience depression based on demographic, academic, lifestyle, and medical history factors. We use a deep learning model to identify hidden patterns in survey data and deploy it via a Streamlit web app for real-time predictions hosted on AWS.
2. Problem Statement
The objective is to build a predictive model that classifies individuals into high-risk or low-risk categories for depression. The model must ensure fairness across demographics (gender, age, profession, city), achieve high predictive accuracy using balanced datasets, and be practical for healthcare, corporate, and governmental decision-making.
3. Dataset
Source: Mental Health Survey Data (CSV format)
Features:
• Demographics: Age, Gender, City
• Academic & Work: Degree, Profession, Work/Study status
• Pressures: Academic, Work, Financial Stress
• Lifestyle: Dietary habits, Sleep duration, Work/Study hours
• History: Family history of mental illness, Suicidal thoughts
• Target Variable: Depression (0 = No, 1 = Yes)
4. Data Preprocessing
The following preprocessing steps were applied on training & test datasets:
• Cleaning & Standardization:
o Corrected invalid entries in categorical fields (City, Profession, Degree, Sleep Duration, Dietary Habits).
o Removed irrelevant columns like Name and ID.
• Handling Missing Values:
o Median imputation for numeric fields.
o Mode imputation for categorical fields.
• Encoding & Transformation:
o One-hot encoding for categorical variables.
o Log transformation for skewed features (Academic Pressure, CGPA, Study Satisfaction).
• Balancing:
o Applied SMOTE stratified by Age Group × Profession to handle class imbalance.
• Scaling:
o StandardScaler used for numerical feature scaling.
5. Model Development
Framework: TensorFlow (Keras API)
Architecture:
• Input layer: 100+ encoded features
• Hidden Layers: Dense layers (256 → 128 → 64 → 32) with ReLU activation, Batch Normalization, Dropout (0.15–0.3)
• Output Layer: 1 neuron with sigmoid activation (binary classification)
Training:
• Loss: Binary Crossentropy
• Optimizer: Adam (lr=0.001)
• Metrics: Accuracy, Precision, Recall, AUC
• Early Stopping: patience = 10
6. Model Evaluation
6.1 Overall Performance Metrics
Validation Metrics (23,999 samples):
• Accuracy: 97%
• Precision: 0.97 (Class 0), 0.97 (Class 1)
• Recall: 0.97 (Class 0), 0.96 (Class 1)
• F1-score: 0.97 (Class 0), 0.96 (Class 1)
• AUC: 0.9946
• Confusion Matrix: [, ]
Test Metrics (48,728 samples):
• Accuracy: 96%
• Precision: 0.96 (Class 0), 0.96 (Class 1)
• Recall: 0.96 (Class 0), 0.96 (Class 1)
• F1-score: 0.96 (Class 0), 0.96 (Class 1)
• AUC: 0.9940
• Confusion Matrix: [, ]
6.2 Demographic Fairness Analysis
• Gender-based Performance:
o Female (24,708 samples): Accuracy 97%, Precision 0.96-0.97, Recall 0.96-0.97
o Male (28,913 samples): Accuracy 96%, Precision 0.96-0.97, Recall 0.96-0.97
• Age Group Performance:
o <18 (839 samples): Accuracy 86%, F1-score 0.77-0.90
o 18-29 (8,325 samples): Accuracy 86%, F1-score 0.86-0.87
o 30-44 (17,170 samples): Accuracy 97%, F1-score 0.97
o 45-59 (22,394 samples): Accuracy 100%, F1-score 1.00
6.3 Professional Category Analysis
• Student Population (7,047 samples):
o Accuracy: 87%
o Precision: 0.90 (Class 0), 0.84 (Class 1)
o F1-score: 0.86-0.87
• Working Professionals (41,681 samples):
o Accuracy: 98%
o Precision: 0.97 (Class 0), 0.99 (Class 1)
o F1-score: 0.98
6.4 Profession-Specific Performance
• High-performing professions (>98% accuracy):
o Digital Marketer: 100% accuracy
o Teacher (11,914 samples): 99% accuracy, F1-score 0.99
o Pharmacist: 99% accuracy
o HR Manager: 99% accuracy
o Chef, Electrician, Manager: 99% accuracy each
• Moderate-performing professions:
o Student: 87% accuracy (largest challenging group)
o Software Engineer: 97% accuracy
o Content Writer: 99% accuracy
6.5 Educational Background Analysis
Performance across degree types shows consistent high accuracy (95-100%):
• PhD, ME, MED: 98-99% accuracy
• MBBS, MD: 97% accuracy
• Engineering degrees (BE, BTECH): 97% accuracy
• Business degrees (BBA, MBA): 97% accuracy
• CLASS 12 (7,143 samples): 95% accuracy
6.6 Geographic Distribution Analysis
City-wise performance demonstrates model robustness:
• Major metros: 97-98% accuracy (Mumbai, Delhi, Bangalore, Chennai)
• Tier-2 cities: 96-98% accuracy (Pune, Ahmedabad, Hyderabad)
• Smaller cities: 96-98% accuracy maintained
7. Deployment
Streamlit Application:
• Mental Health Assessment Page: Users can input details to get risk predictions.
• Business Use Cases Page: Modules for Healthcare Providers, Mental Health Clinics, Corporate Wellness Programs, and Government/NGOs.
Deployment Environment:
• Deployed on AWS (Elastic Beanstalk/EC2)
• Ensures scalability, reliability, and security.
8. Results & Insights
8.1 Model Performance Highlights
• Exceptional Overall Performance: 96-97% accuracy
• Demographic Fairness: Consistent across gender and professions
• Age Patterns: Higher accuracy in middle-aged groups (30-59 years)
• Student Insights: Lower performance highlights need for interventions
8.2 Clinical and Practical Applications
• Healthcare Providers: Early screening with 96% accuracy
• Mental Health Clinics: Queue prioritization with demographic-aware predictions
• Corporate Wellness: Employee monitoring
• Educational Institutions: Targeted student support
• Government & NGOs: Population-level risk analytics
8.3 Model Reliability
• Consistent Cross-validation: 97% vs 96%
• Balanced Precision/Recall
• Demographic Robustness: Stable across 30+ professions and 25+ cities
9. Deliverables
• Source Code: Preprocessing, training, and deployment scripts
• Trained Model: mental_health_survey_final.keras, scaler, and feature columns
• Streamlit App: Interactive web application with demographic-aware predictions
• Comprehensive Evaluation: Metrics across demographics, professions, and locations
• AWS Deployment: Hosted app with enterprise-grade reliability
10. Future Enhancements
• Longitudinal Analysis: Track accuracy over time
• Regional Customization: City-specific fine-tuning
• Professional Risk Profiles: Occupation-based strategies
• Mobile Integration: Smartphone app for accessibility
