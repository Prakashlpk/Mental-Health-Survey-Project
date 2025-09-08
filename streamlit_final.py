import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
from joblib import load
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Mental Health Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and scaler once globally
@st.cache_resource
def load_model_and_scaler():
    try:
        model = keras.models.load_model(r'D:\GUVI\Mental_health_survey\mental_health_survey_final.keras')
        scaler = load(r"D:\GUVI\Mental_health_survey\scaler.save")
        feature_cols = load(r"D:\GUVI\Mental_health_survey\feature_cols.save")
        return model, scaler, feature_cols
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, scaler, feature_cols = load_model_and_scaler()

# Load test data for business use cases
@st.cache_data
def load_and_preprocess_test_data():
    try:
        df_test = pd.read_csv(r'D:\GUVI\Mental_health_survey\final_test_new.zip')
        return df_test
    except Exception as e:
        st.error(f"Error loading test data: {e}")
        return None

test_data = load_and_preprocess_test_data()

# Define helper functions shared for business use cases
def predict_test_batch(df_subset):
    try:
        df_aligned = df_subset.reindex(columns=feature_cols, fill_value=0)
        if "age_group" in df_aligned.columns:
            df_aligned = df_aligned.drop(columns=["age_group"])
        input_scaled = scaler.transform(df_aligned)
        pred_proba = model.predict(input_scaled, verbose=0)
        return pred_proba.flatten()
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return []

def add_business_columns(df):
    df_copy = df.copy()
    np.random.seed(42)
    df_copy['wait_days'] = np.random.randint(1, 30, len(df_copy))
    regions = ["North", "South", "East", "West", "Central"]
    df_copy['region'] = np.random.choice(regions, len(df_copy))
    return df_copy

# Setup navigation tabs or radio buttons
page = st.sidebar.radio("Select Page", ["Mental Health Assessment", "Business Use Cases"])

if page == "Mental Health Assessment":
    # Inject CSS for styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: 700;
            text-align: center;
            color: #2E86AB;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .sub-header {
            font-size: 1.2rem;
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
            font-style: italic;
        }
        .feature-section {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .prediction-box {
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin: 2rem 0;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        }
        .positive-prediction {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .negative-prediction {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        .disclaimer {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
            color: #856404;
        }
        .stButton > button {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.5rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">🧠 Mental Health Assessment Tool</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Depression Risk Prediction</p>', unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Medical Disclaimer:</strong> This tool is for educational and research purposes only. 
        It is not a substitute for professional medical advice, diagnosis, or treatment. 
        Please consult with qualified healthcare professionals for mental health concerns.
    </div>
    """, unsafe_allow_html=True)


    # Halt if model not loaded
    if model is None or scaler is None or feature_cols is None:
        st.error("Model could not be loaded. Please ensure the model files are in the correct directory.")
        st.stop()

    # Input form code as in first script (Personal Info, Academic, Lifestyle, Family History)
    st.markdown('<div class="feature-section">', unsafe_allow_html=True)
    st.markdown("### 📝 Personal Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=16, max_value=100, value=25, help="Your current age")
        gender = st.selectbox("Gender", ["Male", "Female"], help="Select your gender")
    with col2:
        city = st.selectbox("City", [
            "Ahmedabad", "Bangalore", "Bhopal", "Chennai", "Delhi", "Ghaziabad", 
            "Hyderabad", "Indore", "Jaipur", "Kalyan", "Kanpur", "Kolkata", 
            "Mumbai", "Nagpur", "Nashik", "Patna", "Pune", "Surat", "Thane", 
            "Unknown", "Vasai-Virar", "Visakhapatnam"
        ], help="Select your city")
        work_status = st.selectbox("Working Professional or Student", 
                                  ["Student", "Working Professional"], 
                                  help="Your current status")
    with col3:
        profession = st.selectbox("Profession", [
            "Accountant", "Architect", "Business Analyst", "Consultant", "Data Scientist",
            "Doctor", "Engineer", "Financial Analyst", "HR Manager", "Lawyer", "Manager",
            "Marketing Manager", "Nurse", "Project Manager", "Researcher", "Sales Executive",
            "Software Engineer", "Student", "Teacher", "Unknown"
        ], help="Select your profession")
        degree = st.selectbox("Degree", [
            "BARCH", "BBA", "BCA", "BCOM", "BED", "BE", "BTECH", "CLASS 12", "LLB", 
            "MBA", "MCA", "MCOM", "ME", "MTECH", "UNKNOWN"
        ], help="Your highest degree")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="feature-section">', unsafe_allow_html=True)
    st.markdown("### 📚 Academic & Work Information")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        academic_pressure = st.slider("Academic Pressure", 0.0, 5.0, 2.5, 0.1, 
                                     help="Rate your academic pressure (0=None, 5=Extreme)")
        work_pressure = st.slider("Work Pressure", 0.0, 5.0, 2.5, 0.1,
                                help="Rate your work pressure (0=None, 5=Extreme)")
    with col2:
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1,
                              help="Your CGPA or equivalent grade")
        study_satisfaction = st.slider("Study Satisfaction", 0.0, 5.0, 3.0, 0.1,
                                      help="How satisfied are you with your studies?")
    with col3:
        job_satisfaction = st.slider("Job Satisfaction", 0.0, 5.0, 3.0, 0.1,
                                    help="How satisfied are you with your job?")
        work_study_hours = st.number_input("Work/Study Hours per week", min_value=0, max_value=100, value=40,
                                          help="Total hours spent on work/study per week")
    with col4:
        financial_stress = st.slider("Financial Stress", 0.0, 5.0, 2.5, 0.1,
                                    help="Rate your financial stress (0=None, 5=Extreme)")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="feature-section">', unsafe_allow_html=True)
    st.markdown("### 🏃‍♀️ Lifestyle & Health")
    col1, col2, col3 = st.columns(3)
    with col1:
        dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy", "Unknown"],
                                    help="How would you rate your dietary habits?")
    with col2:
        sleep_duration = st.selectbox("Sleep Duration", [
            "<5 hours", "5-6 hours", "6-7 hours", "6-8 hours", 
            "7-8 hours", "8-9 hours", "9-11 hours", "Unknown"
        ], index=4, help="Average hours of sleep per night")
    with col3:
        suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", 
                                        ["No", "Yes"], 
                                        help="This information helps in risk assessment")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="feature-section">', unsafe_allow_html=True)
    st.markdown("### 👨‍👩‍👧‍👦 Family History")
    family_history = st.selectbox("Family History of Mental Illness", ["No", "Yes"],
                                help="Do you have a family history of mental illness?")
    st.markdown('</div>', unsafe_allow_html=True)

    # Prediction button and logic
    st.markdown("### 🔮 Get Your Assessment")
    if st.button("Predict Depression Risk", key="predict_btn"):
        try:
            input_data = {
                'Age': age,
                'Academic Pressure': academic_pressure,
                'Work Pressure': work_pressure,
                'CGPA': cgpa,
                'Study Satisfaction': study_satisfaction,
                'Job Satisfaction': job_satisfaction,
                'Work/Study Hours': work_study_hours,
                'Financial Stress': financial_stress
            }
            for col in feature_cols:
                if col not in input_data:
                    input_data[col] = 0
            input_data[f'Gender_{gender}'] = 1
            input_data[f'City_{city}'] = 1
            input_data[f'Working Professional or Student_{work_status}'] = 1
            input_data[f'Profession_{profession}'] = 1
            input_data[f'Dietary Habits_{dietary_habits}'] = 1
            input_data[f'Degree_{degree}'] = 1
            input_data[f'Have you ever had suicidal thoughts ?_{suicidal_thoughts}'] = 1
            input_data[f'Family History of Mental Illness_{family_history}'] = 1
            input_data[f'Sleep Duration_{sleep_duration}'] = 1

            input_df = pd.DataFrame([input_data])
            input_df = input_df.reindex(columns=feature_cols, fill_value=0)

            if "age_group" in input_df.columns:
                input_df = input_df.drop(columns=["age_group"])

            for col in ['Academic Pressure', 'CGPA', 'Study Satisfaction']:
                input_df[col] = np.log1p(input_df[col])

            input_scaled = scaler.transform(input_df)
            prediction_proba = model.predict(input_scaled, verbose=0)[0][0]
            prediction = int(prediction_proba > 0.5)

            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box negative-prediction">
                    <h2>⚠️ Higher Risk Detected</h2>
                    <p style="font-size: 1.2rem;">Confidence: {prediction_proba:.1%}</p>
                    <p>The model indicates a higher likelihood of depression risk based on the provided information.</p>
                    <p><strong>Recommendation:</strong> Consider speaking with a mental health professional for proper evaluation and support.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box positive-prediction">
                    <h2>✅ Lower Risk Detected</h2>
                    <p style="font-size: 1.2rem;">Confidence: {(1-prediction_proba):.1%}</p>
                    <p>The model indicates a lower likelihood of depression risk based on the provided information.</p>
                    <p><strong>Note:</strong> Continue maintaining good mental health practices and seek help if you experience concerning symptoms.</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 📊 Risk Factors Analysis")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Stress Indicators")
                stress_factors = []
                if academic_pressure > 3.5:
                    stress_factors.append("High Academic Pressure")
                if work_pressure > 3.5:
                    stress_factors.append("High Work Pressure")
                if financial_stress > 3.5:
                    stress_factors.append("High Financial Stress")
                if stress_factors:
                    for factor in stress_factors:
                        st.warning(f"⚠️ {factor}")
                else:
                    st.success("✅ Stress levels appear manageable")
            with col2:
                st.markdown("#### Protective Factors")
                protective_factors = []
                if job_satisfaction > 3.5:
                    protective_factors.append("Good Job Satisfaction")
                if study_satisfaction > 3.5:
                    protective_factors.append("Good Study Satisfaction")
                if dietary_habits == "Healthy":
                    protective_factors.append("Healthy Diet")
                if sleep_duration in ["7-8 hours", "8-9 hours"]:
                    protective_factors.append("Adequate Sleep")
                if protective_factors:
                    for factor in protective_factors:
                        st.success(f"✅ {factor}")
                else:
                    st.info("💡 Consider improving lifestyle factors")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.error("Please ensure all model files are present and try again.")

    # Sidebar content for resources etc.
    st.sidebar.markdown("### 🆘 Mental Health Resources")
    st.sidebar.markdown("""
    **Professional Help:**
    - Consult a psychiatrist or psychologist
    - Visit your local mental health clinic
    - Contact your primary care physician

    **Self-Care Tips:**
    - Maintain regular sleep schedule
    - Exercise regularly
    - Practice mindfulness/meditation
    - Stay connected with supportive people
    - Limit alcohol and substance use
    """)

    # Model Information Expander
    with st.expander("ℹ️ About This Model"):
        st.markdown("""
        **Model Details:**
        - Architecture: Deep Neural Network with 4 hidden layers
        - Features: 19 input features including demographics, lifestyle, and health factors
        - Training: Trained on mental health survey data with SMOTE balancing
        - Performance: Optimized for balanced precision and recall
        
        **Input Features:**
        1. Demographics: Age, Gender, City
        2. Education/Work: Degree, Profession, Work Status
        3. Pressure Scores: Academic, Work, Financial stress levels
        4. Satisfaction: Job and Study satisfaction ratings
        5. Lifestyle: Sleep duration, Dietary habits, Work/Study hours
        6. History: Personal and family mental health history
        
        **Important Notes:**
        - This is a screening tool, not a diagnostic instrument
        - Results should be interpreted by qualified professionals
        - Model accuracy varies across different demographic groups
        """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Developed for educational purposes | Mental Health Awareness 2025</p>
        <p><em>Remember: Seeking help is a sign of strength, not weakness</em></p>
    </div>
    """, unsafe_allow_html=True)


elif page == "Business Use Cases":
    # Halt if model or data cannot load
    if model is None or scaler is None or feature_cols is None or test_data is None:
        st.error("Model or test data could not be loaded. Ensure all required files are available.")
        st.stop()

    # Sidebar for business modules
    module = st.sidebar.selectbox("Select Business Module",
                                 ["Healthcare Providers", "Mental Health Clinics", "Corporate Wellness Programs", "Government and NGOs"])

    st.title("Mental Health Survey - Business Applications")
    st.write(f"**Selected Module:** {module}")
    st.write(f"**Test Dataset Size:** {len(test_data)} records")

    # Business Modules logic as per second script, unchanged for brevity
    if module == "Healthcare Providers":
        st.header("🏥 Healthcare Providers - Early Depression Screening")
        sample_size = st.slider("Select sample size for screening", 50, min(1000, len(test_data)), 200)
        if st.button("Run Screening"):
            with st.spinner("Running batch predictions on test data..."):
                patient_data = test_data.sample(n=sample_size, random_state=42).reset_index(drop=True)
                preds = predict_test_batch(patient_data)
                patient_data['depression_risk'] = preds
                patient_data['risk_category'] = pd.cut(preds, bins=[0, 0.3, 0.7, 1.0], labels=["Low", "Medium", "High"])
            st.markdown("### Risk Distribution")
            risk_counts = patient_data['risk_category'].value_counts()
            st.bar_chart(risk_counts)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Low Risk", risk_counts.get("Low", 0))
            with col2:
                st.metric("Medium Risk", risk_counts.get("Medium", 0))
            with col3:
                st.metric("High Risk", risk_counts.get("High", 0))
            st.markdown("### High-Risk Patients (Sample)")
            high_risk_patients = patient_data[patient_data['risk_category'] == "High"]
            if len(high_risk_patients) > 0:
                display_cols = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Financial Stress', 'depression_risk', 'risk_category']
                available_cols = [col for col in display_cols if col in high_risk_patients.columns]
                st.dataframe(high_risk_patients[available_cols].head(10))
            else:
                st.info("No high-risk patients found in this sample.")

    elif module == "Mental Health Clinics":
        st.header("🧠 Mental Health Clinics - Treatment Prioritization")
        sample_size = st.slider("Select clinic queue size", 20, min(500, len(test_data)), 50)
        if st.button("Generate Priority Queue"):
            with st.spinner("Analyzing patient priority..."):
                clinic_data = test_data.sample(n=sample_size, random_state=42).reset_index(drop=True)
                clinic_data = add_business_columns(clinic_data)
                preds = predict_test_batch(clinic_data)
                clinic_data['depression_risk'] = preds
                clinic_data['priority_score'] = clinic_data['depression_risk'] * 100 + clinic_data['wait_days'] * 0.5
                priority_queue = clinic_data.sort_values('priority_score', ascending=False).reset_index(drop=True)
            st.markdown("### Priority Queue (Top 15)")
            display_cols = ['Age', 'depression_risk', 'wait_days', 'priority_score']
            available_cols = [col for col in display_cols if col in priority_queue.columns]
            st.dataframe(priority_queue[available_cols].head(15))
            st.markdown("### Priority Score Distribution")
            st.line_chart(priority_queue['priority_score'].head(20))

    elif module == "Corporate Wellness Programs":
        st.header("🏢 Corporate Wellness Programs - Employee Mental Health Monitoring")
        sample_size = st.slider("Select employee sample size", 100, min(1000, len(test_data)), 300)
        if st.button("Analyze Employee Mental Health"):
            with st.spinner("Analyzing employee mental health risks..."):
                corporate_data = test_data.sample(n=sample_size, random_state=42).reset_index(drop=True)
                preds = predict_test_batch(corporate_data)
                corporate_data['mental_health_risk'] = preds

            high_risk_count = (corporate_data['mental_health_risk'] > 0.7).sum()
            medium_risk_count = ((corporate_data['mental_health_risk'] > 0.3) & (corporate_data['mental_health_risk'] <= 0.7)).sum()
            low_risk_count = (corporate_data['mental_health_risk'] <= 0.3).sum()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Employees", len(corporate_data))
            with col2:
                st.metric("High Risk", high_risk_count, delta=f"{high_risk_count/len(corporate_data)*100:.1f}%")
            with col3:
                st.metric("Medium Risk", medium_risk_count, delta=f"{medium_risk_count/len(corporate_data)*100:.1f}%")
            with col4:
                st.metric("Low Risk", low_risk_count, delta=f"{low_risk_count/len(corporate_data)*100:.1f}%")
            st.markdown("### Mental Health Risk Distribution")
            fig, ax = plt.subplots()
            ax.hist(corporate_data['mental_health_risk'], bins=20, alpha=0.7, color='steelblue')
            ax.set_xlabel('Mental Health Risk Score')
            ax.set_ylabel('Number of Employees')
            st.pyplot(fig)
            st.markdown("### Risk Categories")
            risk_categories = pd.cut(corporate_data['mental_health_risk'], bins=[0, 0.3, 0.7, 1.0], labels=["Low", "Medium", "High"])
            st.bar_chart(risk_categories.value_counts())

    elif module == "Government and NGOs":
        st.header("🏛️ Government & NGOs - Population Mental Health Analytics")
        sample_size = st.slider("Select population sample size", 200, len(test_data), 500)
        if st.button("Generate Population Analysis"):
            with st.spinner("Analyzing population mental health data..."):
                population_data = test_data.sample(n=sample_size, random_state=42).reset_index(drop=True)
                population_data = add_business_columns(population_data)
                preds = predict_test_batch(population_data)
                population_data['mental_health_risk'] = preds

            st.markdown("### Regional Mental Health Analysis")
            region_summary = population_data.groupby('region').agg({
                'mental_health_risk': ['mean', 'count'],
                'Age': 'mean'
            }).round(3)
            region_summary.columns = ['avg_risk', 'sample_count', 'avg_age']
            region_summary = region_summary.reset_index()
            st.dataframe(region_summary)
            st.markdown("### Average Risk by Region")
            st.bar_chart(region_summary.set_index('region')['avg_risk'])
            st.markdown("### Age Group Risk Analysis")
            age_bins = [0, 25, 35, 45, 55, 100]
            age_labels = ['18-24', '25-34', '35-44', '45-54', '55+']
            population_data['age_group'] = pd.cut(population_data['Age'], bins=age_bins, labels=age_labels, right=False)
            age_risk = population_data.groupby('age_group')['mental_health_risk'].mean()
            st.bar_chart(age_risk)
            st.markdown("### Population Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Risk Score", f"{population_data['mental_health_risk'].mean():.3f}")
            with col2:
                st.metric("High Risk Population", f"{(population_data['mental_health_risk'] > 0.7).sum()}")
            with col3:
                st.metric("Risk Std Deviation", f"{population_data['mental_health_risk'].std():.3f}")

    st.markdown("---")
    st.markdown("**Note:** This application uses preprocessed test data from the mental health survey dataset. Predictions are made using a trained deep learning model.")
