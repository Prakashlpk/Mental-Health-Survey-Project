# Mental Health Survey Project
Deep Learning model to predict depression from survey data
Project Documentation
1. Introduction:
Depression is a growing public health challenge. Early detection can help in timely intervention and treatment. This project aims to predict whether an individual may experience depression based on demographic, academic, lifestyle, and medical history factors. We use a deep learning model to identify hidden patterns in survey data and deploy it via a Streamlit web app for real-time predictions hosted on AWS.
2. Problem Statement:
The objective is to build a predictive model that classifies individuals into high-risk or low-risk categories for depression. The model must ensure fairness across demographics (gender, age, profession, city), achieve high predictive accuracy using balanced datasets, and be practical for healthcare, corporate, and governmental decision-making.
3. Dataset:
- Source: Mental Health Survey Data (CSV format)
- Features:
  - Demographics: Age, Gender, City
  - Academic & Work: Degree, Profession, Work/Study status
  - Pressures: Academic, Work, Financial Stress
  - Lifestyle: Dietary habits, Sleep duration, Work/Study hours
  - History: Family history of mental illness, Suicidal thoughts
- Target Variable: Depression (0 = No, 1 = Yes)
4. Data Preprocessing:
The following preprocessing steps were applied on training & test datasets:

- Cleaning & Standardization:
  - Corrected invalid entries in categorical fields (City, Profession, Degree, Sleep Duration, Dietary Habits).
  - Removed irrelevant columns like Name and ID.

- Handling Missing Values:
  - Median imputation for numeric fields.
  - Mode imputation for categorical fields.

- Encoding & Transformation:
  - One-hot encoding for categorical variables.
  - Log transformation for skewed features (Academic Pressure, CGPA, Study Satisfaction).

- Balancing:
  - Applied SMOTE stratified by Age Group × Profession to handle class imbalance.

- Scaling:
  - StandardScaler used for numerical feature scaling.
5. Model Development:
- Framework: TensorFlow (Keras API)
- Architecture:
  - Input layer: 100+ encoded features
  - Hidden Layers: Dense layers (256 → 128 → 64 → 32) with ReLU activation, Batch Normalization, Dropout (0.15–0.3)
  - Output Layer: 1 neuron with sigmoid activation (binary classification)

- Training:
  - Loss: Binary Crossentropy
  - Optimizer: Adam (lr=0.001)
  - Metrics: Accuracy, Precision, Recall, AUC
  - Early Stopping: patience = 10
6. Model Evaluation:
Validation Metrics:
- Accuracy: ~84%
- Precision: ~0.82
- Recall: ~0.80
- F1-score: ~0.81
- AUC: ~0.88

Test Metrics:
- Accuracy: ~83%
- Precision: ~0.81
- Recall: ~0.79
- F1-score: ~0.80
- AUC: ~0.87

Fairness Analysis:
- Performance was stable across gender, age group, profession, degree, and city categories.
- Minor imbalances mitigated with SMOTE.
7. Deployment
The project has been successfully deployed with the following options:

- Streamlit Application:
  - Mental Health Assessment Page: Users can input personal, academic, lifestyle, and history details to get risk predictions.
  - Business Use Cases Page: Modules for Healthcare Providers, Mental Health Clinics, Corporate Wellness Programs, and Government/NGOs.

- Deployment Environment:
  - The Streamlit app is deployed on AWS (Elastic Beanstalk/EC2) for real-time global access.
  - Ensures scalability, reliability, and security for end users.
8. Results & Insights:
- The model achieved high accuracy and fairness across demographic groups.
- Healthcare Providers: Enables early screening of high-risk patients.
- Mental Health Clinics: Prioritization of treatment queues.
- Corporates: Employee mental health monitoring.
- Governments & NGOs: Regional and population-level risk analytics.
9. Deliverables:
- Source Code: Data preprocessing, model training, and deployment scripts.
- Trained Model: `mental_health_survey_final.keras`, scaler, and feature columns.
- Streamlit App: Interactive web application.
- Documentation: Detailed explanation of approach, data, model, and evaluation results.
- AWS Deployment: Hosted app accessible for real-time predictions.
